// src/supabaseClient.js

import { createClient } from '@supabase/supabase-js';

// Replace with your actual Supabase project URL and anon key
const supabaseUrl = 'https://uzpyjrilollsndgrhqow.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV6cHlqcmlsb2xsc25kZ3JocW93Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NTA2MjUsImV4cCI6MjA2OTQyNjYyNX0.bc0khF7d8rQlQq9kl_1rSu6V0Lkz-abPLbOhNFNXAHw';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
