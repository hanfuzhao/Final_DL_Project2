-- SafeType Parent Dashboard — Supabase Schema
-- Run this in Supabase SQL Editor (https://supabase.com/dashboard -> SQL Editor)

-- Messages table: stores all captured messages from child's device
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    device_id TEXT,
    text TEXT NOT NULL,
    sender TEXT,
    direction TEXT DEFAULT 'unknown',
    app_source TEXT NOT NULL,
    source_layer TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    conversation_hash TEXT,
    is_flagged BOOLEAN DEFAULT NULL,
    flag_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast filtering
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_app_source ON messages(app_source);
CREATE INDEX IF NOT EXISTS idx_messages_is_flagged ON messages(is_flagged);
CREATE INDEX IF NOT EXISTS idx_messages_source_layer ON messages(source_layer);
CREATE INDEX IF NOT EXISTS idx_messages_device_id ON messages(device_id);

-- Enable Row Level Security
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Policy: allow anyone (anon + authenticated) to read messages (dashboard uses anon key)
CREATE POLICY "Anyone can read messages"
    ON messages FOR SELECT
    USING (true);

-- Policy: allow inserts from anon (device uploads with anon key)
CREATE POLICY "Allow device inserts"
    ON messages FOR INSERT
    TO anon
    WITH CHECK (true);

-- Policy: allow updates from anyone (for edge function flagging and dashboard)
CREATE POLICY "Allow message updates"
    ON messages FOR UPDATE
    USING (true)
    WITH CHECK (true);

-- Enable realtime for the messages table
ALTER PUBLICATION supabase_realtime ADD TABLE messages;
