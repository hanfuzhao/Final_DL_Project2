-- Fix RLS policies for SafeType messages table
-- Run this in Supabase SQL Editor to fix the flagging and dashboard access

-- Drop old restrictive policies
DROP POLICY IF EXISTS "Authenticated users can read messages" ON messages;
DROP POLICY IF EXISTS "Authenticated users can update messages" ON messages;
DROP POLICY IF EXISTS "Allow device inserts" ON messages;

-- New policies: allow anon (dashboard + device) to read and update
CREATE POLICY "Anyone can read messages"
    ON messages FOR SELECT
    USING (true);

CREATE POLICY "Allow device inserts"
    ON messages FOR INSERT
    TO anon
    WITH CHECK (true);

CREATE POLICY "Allow message updates"
    ON messages FOR UPDATE
    USING (true)
    WITH CHECK (true);
