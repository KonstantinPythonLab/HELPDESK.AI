/**
 * Modern Multi-Provider Speech-to-Text (STT) Service
 * Supported Providers & Fallback Pipeline:
 * 1. Groq Cloud Whisper API (whisper-large-v3 / whisper-large-v3-turbo) - ultra fast & high accuracy
 * 2. Hugging Face Inference Whisper API (openai/whisper-large-v3-turbo) - free tier open-source
 * 3. Browser Native SpeechRecognition / webkitSpeechRecognition API - zero latency direct dictation
 */

// Helper to convert an Audio Blob into a WAV/WebM File object
export const createAudioFile = (audioBlob, filename = 'speech.webm') => {
    return new File([audioBlob], filename, { type: audioBlob.type || 'audio/webm' });
};

/**
 * Transcribe Audio Blob using Groq Whisper API
 */
export const transcribeWithGroq = async (audioBlob, groqApiKey) => {
    if (!groqApiKey) throw new Error("Groq API key not provided");

    const formData = new FormData();
    const file = createAudioFile(audioBlob, 'recording.webm');
    formData.append('file', file);
    formData.append('model', 'whisper-large-v3-turbo');
    formData.append('response_format', 'json');
    formData.append('temperature', '0');

    const res = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${groqApiKey}`
        },
        body: formData
    });

    if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData?.error?.message || `Groq Whisper HTTP ${res.status}`);
    }

    const data = await res.json();
    return data.text || '';
};

/**
 * Transcribe Audio Blob using Hugging Face Free Inference API
 */
export const transcribeWithHuggingFace = async (audioBlob, hfToken) => {
    const headers = { 'Content-Type': audioBlob.type || 'audio/webm' };
    if (hfToken) {
        headers['Authorization'] = `Bearer ${hfToken}`;
    }

    const res = await fetch(
        'https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo',
        {
            method: 'POST',
            headers,
            body: audioBlob
        }
    );

    if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData?.error || `Hugging Face Whisper HTTP ${res.status}`);
    }

    const data = await res.json();
    return data.text || '';
};
