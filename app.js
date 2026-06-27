const issueBox = document.getElementById('issueBox');
const stepsList = document.getElementById('stepsList');
const scanButton = document.getElementById('scanButton');

const diagnostics = {
  issue: 'Likely overheating after sustained use',
  steps: [
    'Pause the app and let the device cool for 2–3 minutes.',
    'Close background apps to reduce thermal load.',
    'If the issue repeats, check for a software update or battery health degradation.'
  ]
};

scanButton.addEventListener('click', () => {
  issueBox.textContent = diagnostics.issue;
  stepsList.innerHTML = diagnostics.steps.map((step) => `<li>${step}</li>`).join('');
});
