# Key exchange example
await client.room_send(
    room_id,
    message_type="halos.encryption_key",
    content={"key": "GawgguFz..."}
)