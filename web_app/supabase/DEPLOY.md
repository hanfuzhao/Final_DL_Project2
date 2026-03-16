# Deploying the Analyze-Messages Edge Function

## Prerequisites
1. Install Supabase CLI: `brew install supabase/tap/supabase`
2. Login: `supabase login`
3. Link to your project: `supabase link --project-ref vdgjozaouhjhjzrwuqvi`

## Deploy the Edge Function
```bash
supabase functions deploy analyze-messages
```

## Fix RLS Policies
Run the SQL in `web/fix_rls_policies.sql` in your Supabase SQL Editor to allow
the dashboard and edge function to read/update messages.

## How It Works
- The edge function reads messages where `is_flagged IS NULL`
- Sends them to the SafeType DistilBERT API at `https://dl-project-2-second-version.onrender.com/api/predict`
- Updates `is_flagged` and `flag_reason` columns in the messages table
- The parent dashboard calls this function automatically every 60 seconds
- You can also click "Analyze Now" on the dashboard to trigger manually
- Clicking any message on the dashboard directly calls the API for detailed 10-category analysis
