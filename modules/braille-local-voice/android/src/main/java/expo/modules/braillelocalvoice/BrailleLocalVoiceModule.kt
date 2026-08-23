package expo.modules.braillelocalvoice

import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import expo.modules.kotlin.Promise
import expo.modules.kotlin.exception.CodedException
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.records.Field
import expo.modules.kotlin.records.Record
import expo.modules.kotlin.types.OptimizedRecord
import java.util.ArrayDeque

private const val TEXT_TO_SPEECH_INITIALIZATION_TIMEOUT_MS = 5_000L

@OptimizedRecord
data class LocalVoiceRecord(
  @Field val identifier: String,
  @Field val language: String,
  @Field val name: String
) : Record

private class LocalVoiceProbeException(message: String, cause: Throwable? = null) :
  CodedException(message, cause)

class BrailleLocalVoiceModule : Module() {
  private val lock = Any()
  private val mainHandler = Handler(Looper.getMainLooper())
  private val pendingProbes = ArrayDeque<Promise>()
  private var textToSpeech: TextToSpeech? = null
  private var initializing = false
  private var ready = false
  private var destroyed = false
  private var activeInitializationToken = 0
  private var initializationTimeout: Runnable? = null

  override fun definition() = ModuleDefinition {
    Name("BrailleLocalVoice")

    AsyncFunction("getLocalVoicesAsync") { promise: Promise ->
      requestLocalVoices(promise)
    }

    OnDestroy {
      shutdown()
    }
  }

  private fun requestLocalVoices(promise: Promise) {
    var readyInstance: TextToSpeech? = null
    var initializationToken: Int? = null

    synchronized(lock) {
      if (destroyed) {
        promise.reject(LocalVoiceProbeException("Probe suara lokal sudah ditutup."))
        return
      }

      val current = textToSpeech
      if (ready && current != null) {
        readyInstance = current
      } else {
        pendingProbes.add(promise)
        if (current == null && !initializing) {
          initializing = true
          activeInitializationToken += 1
          initializationToken = activeInitializationToken
        }
      }
    }

    readyInstance?.let {
      resolveVoices(promise, it)
      return
    }

    initializationToken?.let(::initializeTextToSpeech)
  }

  private fun initializeTextToSpeech(token: Int) {
    if (Looper.myLooper() == Looper.getMainLooper()) {
      initializeTextToSpeechOnMainThread(token)
    } else {
      mainHandler.post { initializeTextToSpeechOnMainThread(token) }
    }
  }

  private fun initializeTextToSpeechOnMainThread(token: Int) {
    val shouldStart = synchronized(lock) {
      !destroyed && initializing && activeInitializationToken == token && textToSpeech == null
    }
    if (!shouldStart) {
      return
    }

    val context = appContext.reactContext?.applicationContext
    if (context == null) {
      rejectPending("Konteks Android untuk memeriksa suara lokal tidak tersedia.")
      return
    }

    try {
      val instance = TextToSpeech(context) { status ->
        // Posting memastikan constructor selesai dan instance sudah tersimpan.
        mainHandler.post { handleInitialization(token, status) }
      }
      val shouldKeep = synchronized(lock) {
        if (!destroyed && initializing && activeInitializationToken == token) {
          textToSpeech = instance
          true
        } else {
          false
        }
      }
      if (!shouldKeep) {
        instance.shutdown()
        return
      }
      scheduleInitializationTimeout(token)
    } catch (error: Throwable) {
      rejectPending("Text-to-Speech Android tidak dapat diinisialisasi.", error)
    }
  }

  private fun handleInitialization(token: Int, status: Int) {
    val instance: TextToSpeech?
    val probes: List<Promise>
    val timeout: Runnable?

    synchronized(lock) {
      if (!initializing || activeInitializationToken != token) {
        return
      }
      timeout = initializationTimeout
      initializationTimeout = null
      instance = textToSpeech
      if (!destroyed && status == TextToSpeech.SUCCESS && instance != null) {
        initializing = false
        ready = true
        probes = pendingProbes.toList()
        pendingProbes.clear()
      } else {
        initializing = false
        ready = false
        textToSpeech = null
        probes = pendingProbes.toList()
        pendingProbes.clear()
      }
    }
    timeout?.let(mainHandler::removeCallbacks)

    if (status == TextToSpeech.SUCCESS && instance != null && !destroyed) {
      probes.forEach { resolveVoices(it, instance) }
      return
    }

    instance?.shutdown()
    probes.forEach {
      it.reject(LocalVoiceProbeException("Text-to-Speech Android gagal diinisialisasi."))
    }
  }

  private fun scheduleInitializationTimeout(token: Int) {
    val timeout = Runnable { handleInitializationTimeout(token) }
    val previous = synchronized(lock) {
      if (!initializing || activeInitializationToken != token) {
        return
      }
      val current = initializationTimeout
      initializationTimeout = timeout
      current
    }
    previous?.let(mainHandler::removeCallbacks)
    mainHandler.postDelayed(timeout, TEXT_TO_SPEECH_INITIALIZATION_TIMEOUT_MS)
  }

  private fun handleInitializationTimeout(token: Int) {
    val timedOut = synchronized(lock) {
      initializing && activeInitializationToken == token && !destroyed
    }
    if (timedOut) {
      rejectPending("Text-to-Speech Android melewati batas waktu inisialisasi.")
    }
  }

  private fun resolveVoices(promise: Promise, instance: TextToSpeech) {
    try {
      val voices = instance.voices
        .orEmpty()
        .asSequence()
        .filterNot(Voice::isNetworkConnectionRequired)
        .map {
          LocalVoiceRecord(
            identifier = it.name,
            language = it.locale.toLanguageTag(),
            name = it.name
          )
        }
        .sortedWith(
          compareByDescending<LocalVoiceRecord> { it.language.equals("id-ID", ignoreCase = true) }
            .thenByDescending { it.language.startsWith("id", ignoreCase = true) }
            .thenBy { it.identifier }
        )
        .toList()
      promise.resolve(voices)
    } catch (error: Throwable) {
      promise.reject(LocalVoiceProbeException("Daftar suara lokal tidak dapat diperiksa.", error))
    }
  }

  private fun rejectPending(message: String, cause: Throwable? = null) {
    val probes: List<Promise>
    val instance: TextToSpeech?
    val timeout: Runnable?
    synchronized(lock) {
      initializing = false
      ready = false
      timeout = initializationTimeout
      initializationTimeout = null
      instance = textToSpeech
      textToSpeech = null
      probes = pendingProbes.toList()
      pendingProbes.clear()
    }
    timeout?.let(mainHandler::removeCallbacks)
    instance?.shutdown()
    probes.forEach { it.reject(LocalVoiceProbeException(message, cause)) }
  }

  private fun shutdown() {
    val probes: List<Promise>
    val instance: TextToSpeech?
    val timeout: Runnable?
    synchronized(lock) {
      if (destroyed) {
        return
      }
      destroyed = true
      initializing = false
      ready = false
      timeout = initializationTimeout
      initializationTimeout = null
      instance = textToSpeech
      textToSpeech = null
      probes = pendingProbes.toList()
      pendingProbes.clear()
    }

    timeout?.let(mainHandler::removeCallbacks)
    instance?.shutdown()
    probes.forEach {
      it.reject(LocalVoiceProbeException("Probe suara lokal ditutup sebelum selesai."))
    }
  }
}
