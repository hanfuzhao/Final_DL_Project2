package com.safetype.keyboard.api

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * API client factory.
 *
 * Builds Retrofit instance with:
 * - JWT auth interceptor (token from EncryptedSharedPreferences)
 * - Logging interceptor for debug builds
 * - Configurable base URL (default: Supabase Edge Functions or Railway)
 */
object ApiClient {

    private const val PREFS_FILE = "safetype_secure_prefs"
    private const val KEY_JWT_TOKEN = "jwt_token"
    private const val KEY_BASE_URL = "api_base_url"

    // Default base URL — update this to your Railway/Render endpoint
    private const val DEFAULT_BASE_URL = "https://vdgjozaouhjhjzrwuqvi.supabase.co/functions/v1/"

    @Volatile
    private var instance: ApiService? = null

    fun getInstance(context: Context): ApiService {
        return instance ?: synchronized(this) {
            instance ?: buildApiService(context).also { instance = it }
        }
    }

    private fun buildApiService(context: Context): ApiService {
        val prefs = getEncryptedPrefs(context)
        val baseUrl = prefs.getString(KEY_BASE_URL, DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL

        val authInterceptor = Interceptor { chain ->
            val token = getEncryptedPrefs(context).getString(KEY_JWT_TOKEN, null)
            val request = if (token != null) {
                chain.request().newBuilder()
                    .addHeader("Authorization", "Bearer $token")
                    .build()
            } else {
                chain.request()
            }
            chain.proceed(request)
        }

        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }

        val client = OkHttpClient.Builder()
            .addInterceptor(authInterceptor)
            .addInterceptor(logging)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()

        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }

    // ─── Token management ───

    fun saveToken(context: Context, token: String) {
        getEncryptedPrefs(context).edit().putString(KEY_JWT_TOKEN, token).apply()
        // Reset instance to pick up new token
        instance = null
    }

    fun getToken(context: Context): String? {
        return getEncryptedPrefs(context).getString(KEY_JWT_TOKEN, null)
    }

    fun hasToken(context: Context): Boolean {
        return getToken(context) != null
    }

    fun setBaseUrl(context: Context, url: String) {
        getEncryptedPrefs(context).edit().putString(KEY_BASE_URL, url).apply()
        instance = null
    }

    private fun getEncryptedPrefs(context: Context) =
        EncryptedSharedPreferences.create(
            PREFS_FILE,
            MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC),
            context,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
}
