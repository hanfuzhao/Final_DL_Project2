package com.safetype.keyboard.data

/**
 * Supabase connection config.
 * Using anon key for device uploads — RLS policies control access.
 * Threat analysis is handled server-side by the Supabase Edge Function.
 */
object SupabaseConfig {
    const val URL = "https://vdgjozaouhjhjzrwuqvi.supabase.co"
    const val ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZkZ2pvemFvdWhqaGp6cnd1cXZpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMzMjc5MDYsImV4cCI6MjA4ODkwMzkwNn0.CwWRWedncN1aBceCJctmPDc9Azzo1TT64sDqOxmdm9o"
}
