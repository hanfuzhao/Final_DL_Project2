/**
 * Supabase configuration — credentials are baked in.
 */
const SUPABASE_URL = 'https://vdgjozaouhjhjzrwuqvi.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZkZ2pvemFvdWhqaGp6cnd1cXZpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMzMjc5MDYsImV4cCI6MjA4ODkwMzkwNn0.CwWRWedncN1aBceCJctmPDc9Azzo1TT64sDqOxmdm9o';

const SafeTypeConfig = {
    getSupabaseClient() {
        return supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    },

    getSupabaseUrl() {
        return SUPABASE_URL;
    },

    getAnonKey() {
        return SUPABASE_ANON_KEY;
    }
};
