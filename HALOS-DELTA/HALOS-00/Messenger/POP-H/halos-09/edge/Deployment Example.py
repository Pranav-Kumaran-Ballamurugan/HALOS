# Example workflow
async def demo():
    # Start ephemeral voice huddle
    huddle = VoiceHuddle()
    await huddle.start_huddle(["alice", "bob"])
    
    # Simulate audio processing
    await huddle.transcribe_segment(b"raw_audio_data...")
    
    # End and summarize
    summary = await huddle.end_huddle()
    print(f"Meeting summary: {summary}")
    
    # Create MPC budget pool
    mpc = MPCBudget(["alice", "bob"])
    alice_commit = mpc.add_commitment("alice", max_amount=1000)
    bob_commit = mpc.add_commitment("bob", max_amount=800)
    
    # Calculate fair split for $1200 expense
    splits = mpc.compute_optimal_split(1200)
    print(f"Fair splits: {splits}")  # e.g. {"alice": 720, "bob": 480}