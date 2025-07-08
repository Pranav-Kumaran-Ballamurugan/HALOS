<!-- src/Chat.svelte -->
<script>
  let messages = [];
  let input = "";
  
  async function send() {
    await messenger.send(roomId, input);
    input = "";
  }
</script>

<div class="chat-container">
  {#each messages as msg}
    <div class="message {msg.sender === me ? 'sent' : 'received'}">
      {msg.text}
    </div>
  {/each}
  
  <input bind:value={input} on:keydown={e => e.key === 'Enter' && send()}>
  <button on:click={send}>Send</button>
</div>