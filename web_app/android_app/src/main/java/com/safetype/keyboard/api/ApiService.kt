package com.safetype.keyboard.api

import com.google.gson.annotations.SerializedName
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

/**
 * Retrofit API interface for the SafeType backend.
 *
 * Two endpoints:
 * 1. /api/v1/analyze — batch upload messages for analysis/flagging
 * 2. /api/v1/pair   — pair device with parent dashboard
 *
 * JWT auth token is injected via AuthInterceptor.
 */
interface ApiService {

    @POST("api/v1/analyze")
    suspend fun analyzeMessages(@Body request: AnalyzeRequest): Response<AnalyzeResponse>

    @POST("api/v1/pair")
    suspend fun pairDevice(@Body request: PairRequest): Response<PairResponse>
}

// ─── Request/Response Models ───

data class AnalyzeRequest(
    @SerializedName("device_id") val deviceId: String,
    val messages: List<MessagePayload>
)

data class MessagePayload(
    val id: Long,
    @SerializedName("app_source") val appSource: String,
    val sender: String?,
    val text: String,
    val direction: String,
    @SerializedName("conversation_hash") val conversationHash: String?,
    val timestamp: Long,
    val context: List<ContextMessage>? = null
)

data class ContextMessage(
    val text: String,
    val sender: String?,
    val timestamp: Long
)

data class AnalyzeResponse(
    val status: String,
    val flagged: List<FlaggedItem>? = null
)

data class FlaggedItem(
    val id: Long,
    val reason: String,
    val severity: String
)

data class PairRequest(
    @SerializedName("pairing_code") val pairingCode: String,
    @SerializedName("device_model") val deviceModel: String,
    @SerializedName("device_name") val deviceName: String
)

data class PairResponse(
    val status: String,
    val token: String? = null,
    val message: String? = null
)
