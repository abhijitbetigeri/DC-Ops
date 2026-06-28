// Dependency-free TCP relay that runs the YOLO QNN backbone on the Hexagon NPU.
// Runs as the `shell` user from /data/local/tmp, where the cDSP grants an unsigned-PD
// session (a normal app cannot). The DC-Ops app connects over 127.0.0.1 and streams
// frames; this process shells out to the proven qnn_executor_runner per frame (~20ms).
//
// Protocol (fixed sizes, no headers — both sides know the shapes):
//   client -> server:  IN_BYTES  bytes  (1x3x640x640 float32, NCHW, /255)
//   server -> client:  OUT_BYTES bytes  (5 tensors concatenated, see OUT_SIZES)
//
// Build: aarch64-linux-android26-clang yolo_npu_server.c -o yolo_npu_server -O2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define PORT 8765
#define WS "/data/local/tmp/dcops_yolo_qnn"
#define IN_BYTES (1*3*640*640*4)
static const long OUT_SIZES[5] = {
    1L*80*80*80*4,   // output_0_0  ps0
    1L*80*40*40*4,   // output_0_1  ps1
    1L*80*20*20*4,   // output_0_2  ps2
    1L*32*8400*4,    // output_0_3  mc
    1L*32*160*160*4, // output_0_4  proto
};

static int read_full(int fd, char *buf, long n) {
    long got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += r;
    }
    return 0;
}
static int write_full(int fd, const char *buf, long n) {
    long sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, buf + sent, n - sent);
        if (w <= 0) return -1;
        sent += w;
    }
    return 0;
}
static long read_file(const char *path, char *buf, long cap) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    long got = 0; ssize_t r;
    while (got < cap && (r = read(fd, buf + got, cap - got)) > 0) got += r;
    close(fd);
    return got;
}

// Run the runner; returns 0 on success.
static int run_inference(void) {
    pid_t pid = fork();
    if (pid == 0) {
        // child: cd WS, exec the runner with the right env
        chdir(WS);
        setenv("LD_LIBRARY_PATH", ".", 1);
        setenv("ADSP_LIBRARY_PATH", ".", 1);
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, 1); dup2(devnull, 2);
        execl("./qnn_executor_runner", "./qnn_executor_runner",
              "--model_path", "model.pte",
              "--input_list_path", "input_list.txt",
              "--output_folder_path", "out", (char*)NULL);
        _exit(127);
    }
    if (pid < 0) return -1;
    int st; waitpid(pid, &st, 0);
    return (WIFEXITED(st) && WEXITSTATUS(st) == 0) ? 0 : -1;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGCHLD, SIG_DFL);

    char *inbuf  = malloc(IN_BYTES);
    long out_total = 0;
    for (int i = 0; i < 5; i++) out_total += OUT_SIZES[i];
    char *outbuf = malloc(out_total);
    if (!inbuf || !outbuf) { fprintf(stderr, "OOM\n"); return 1; }

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // 127.0.0.1 only
    addr.sin_port = htons(PORT);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "bind failed: %s\n", strerror(errno)); return 1;
    }
    listen(srv, 4);
    fprintf(stderr, "yolo_npu_server: listening on 127.0.0.1:%d, ws=%s\n", PORT, WS);
    fflush(stderr);

    for (;;) {
        int cli = accept(srv, NULL, NULL);
        if (cli < 0) continue;
        int nodelay = 1;
        setsockopt(cli, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        fprintf(stderr, "client connected\n"); fflush(stderr);

        // Serve frames on this connection until the client disconnects.
        while (read_full(cli, inbuf, IN_BYTES) == 0) {
            // write input.raw
            int fd = open(WS "/input.raw", O_WRONLY|O_CREAT|O_TRUNC, 0644);
            if (fd < 0 || write_full(fd, inbuf, IN_BYTES) != 0) { if(fd>=0)close(fd); break; }
            close(fd);

            if (run_inference() != 0) { fprintf(stderr, "inference failed\n"); break; }

            // gather the 5 outputs in order
            long off = 0; int ok = 1;
            for (int i = 0; i < 5; i++) {
                char p[128]; snprintf(p, sizeof(p), WS "/out/output_0_%d.raw", i);
                long got = read_file(p, outbuf + off, OUT_SIZES[i]);
                if (got != OUT_SIZES[i]) { fprintf(stderr, "bad out %d got=%ld want=%ld\n", i, got, OUT_SIZES[i]); ok = 0; break; }
                off += OUT_SIZES[i];
            }
            if (!ok) break;
            if (write_full(cli, outbuf, out_total) != 0) break;
        }
        close(cli);
        fprintf(stderr, "client disconnected\n"); fflush(stderr);
    }
    return 0;
}
