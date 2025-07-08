// src/components/CampaignCreator.svelte
<script>
  let campaign = {
    goal: 1200,
    members: [],
    deadline: new Date(+new Date() + 30*86400e3)
  };

  function addMember() {
    campaign.members = [...campaign.members, {id: "", amount: 0}];
  }
</script>

<article class="campaign-card">
  <h2>New Group Campaign</h2>
  
  <input bind:value={campaign.goal} type="number" placeholder="Target amount ($)">
  
  {#each campaign.members as member}
    <div class="member-row">
      <input bind:value={member.id} placeholder="HALOS ID">
      <input bind:value={member.amount} type="number" placeholder="$">
    </div>
  {/each}
  
  <button on:click={addMember}>+ Add Member</button>
</article>