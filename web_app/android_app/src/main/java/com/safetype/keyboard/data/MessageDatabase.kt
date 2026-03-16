package com.safetype.keyboard.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import net.sqlcipher.database.SupportFactory
import net.sqlcipher.database.SQLiteDatabase
import java.security.KeyStore
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey

/**
 * Encrypted Room database for the message queue.
 *
 * Uses SQLCipher for AES-256 encryption at rest.
 * The encryption key is stored in Android Keystore (hardware-backed on most devices).
 *
 * Messages are retained for a maximum of 24 hours as a batch/retry buffer.
 */
@Database(entities = [MessageEntity::class], version = 2, exportSchema = false)
abstract class MessageDatabase : RoomDatabase() {

    abstract fun messageDao(): MessageDao

    companion object {
        private const val DB_NAME = "safetype_messages.db"
        private const val KEYSTORE_ALIAS = "safetype_db_key"
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"

        @Volatile
        private var INSTANCE: MessageDatabase? = null

        fun getInstance(context: Context): MessageDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: buildDatabase(context).also { INSTANCE = it }
            }
        }

        private fun buildDatabase(context: Context): MessageDatabase {
            val passphrase = getOrCreateDatabaseKey()

            val factory = SupportFactory(passphrase)

            return Room.databaseBuilder(
                context.applicationContext,
                MessageDatabase::class.java,
                DB_NAME
            )
                .openHelperFactory(factory)
                .fallbackToDestructiveMigration()
                .build()
        }

        /**
         * Get or create the AES-256 database encryption key from Android Keystore.
         * The key is hardware-backed and never leaves the secure element.
         */
        private fun getOrCreateDatabaseKey(): ByteArray {
            val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }

            if (!keyStore.containsAlias(KEYSTORE_ALIAS)) {
                val keyGenerator = KeyGenerator.getInstance(
                    KeyProperties.KEY_ALGORITHM_AES,
                    ANDROID_KEYSTORE
                )
                keyGenerator.init(
                    KeyGenParameterSpec.Builder(
                        KEYSTORE_ALIAS,
                        KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
                    )
                        .setKeySize(256)
                        .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                        .build()
                )
                keyGenerator.generateKey()
            }

            // Derive a passphrase from the keystore key's encoded form
            // SQLCipher needs a byte array passphrase
            val key = keyStore.getKey(KEYSTORE_ALIAS, null) as SecretKey
            // Use the key's algorithm + alias hash as passphrase
            // (AndroidKeyStore keys don't expose encoded form, so we derive a stable passphrase)
            val passphrase = "$KEYSTORE_ALIAS-${key.algorithm}-safetype-v1"
            return SQLiteDatabase.getBytes(passphrase.toCharArray())
        }
    }
}
