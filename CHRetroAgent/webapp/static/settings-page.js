SR.init('settings');
let connection = null;
async function loadConnection() {
  connection = await SR.api('/api/llm/status');
  SR.$('#connection-provider').textContent = connection.provider;
  SR.$('#connection-model').textContent = connection.model || 'No model configured';
  SR.$('#connection-badge').textContent = connection.configured ? 'Default connection configured' : 'Backend setup required';
  SR.$('#credential-status').textContent = connection.credential_saved
    ? 'Credential saved on this computer. The workspace connects automatically; no API key entry is needed.'
    : connection.key_available ? 'The server supplies your credential automatically.' : 'A server credential has not been configured yet.';
  SR.$('#llm-url').value = connection.url || '';
  SR.$('#llm-model').value = connection.model || '';
  SR.$('#llm-fallback').checked = !!connection.step_fallback;
  SR.$('#verify-connection').disabled = !connection.configured;
}
async function verify(button, request) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'Verifying…';
  SR.$('#connection-result').textContent = '';
  try {
    const result = await SR.api('/api/llm/configure', {method:'POST',body:JSON.stringify(request)});
    await loadConnection();
    SR.$('#connection-result').textContent = `Connection verified. ${result.model} is ready for Agent conversations.`;
    SR.toast('Default LLM connection verified');
  } catch (error) {
    SR.$('#connection-result').textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}
SR.$('#verify-connection').onclick = event => verify(event.currentTarget, {
  url:connection.url,model:connection.model,step_fallback:connection.step_fallback
});
SR.$('#llm-form').onsubmit = event => {
  event.preventDefault();
  verify(event.submitter, {url:SR.$('#llm-url').value.trim(),model:SR.$('#llm-model').value.trim(),step_fallback:SR.$('#llm-fallback').checked});
};
loadConnection().catch(error => SR.$('#connection-result').textContent = error.message);
