import axios from "axios";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== "undefined" &&
  window.location.hostname !== "localhost" &&
  window.location.hostname !== "127.0.0.1"
    ? "https://support-ticket-sentiment-classifier.onrender.com"
    : "http://127.0.0.1:8000");

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

/**
 * Predict sentiment for given text using the live FastAPI backend ('bert', 'tfidf', or 'both').
 * Never falls back to toy keyword rules; propagates descriptive errors so UI remains truthful.
 */
export async function predictSentiment(text, model = "bert") {
  try {
    const response = await apiClient.post("/api/v1/predict", {
      text,
      model,
    });
    return {
      ...response.data,
      source: response.data?.source || "live-api",
    };
  } catch (error) {
    const serverDetail = error?.response?.data?.detail;
    const statusCode = error?.response?.status;
    if (serverDetail) {
      throw new Error(`Backend Error (${statusCode || "API"}): ${serverDetail}`);
    }
    if (error?.code === "ECONNABORTED") {
      throw new Error(
        `Request to ${API_BASE_URL} timed out after 30s. If using a cold-started server, please retry.`
      );
    }
    throw new Error(
      `Cannot reach sentiment backend at ${API_BASE_URL}. Ensure the FastAPI server is running.`
    );
  }
}

/**
 * Check backend health and individual model loading status.
 */
export async function checkBackendHealth() {
  try {
    const response = await apiClient.get("/api/v1/health");
    return response.data;
  } catch (error) {
    return {
      status: "offline",
      models_loaded: false,
      bert_loaded: false,
      tfidf_loaded: false,
      available_models: [],
      error: error?.message || "Backend unreachable",
    };
  }
}
