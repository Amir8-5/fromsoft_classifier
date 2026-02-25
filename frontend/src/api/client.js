import axios from 'axios';

const apiClient = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
});

/**
 * Send an image file to the FastAPI backend for classification.
 * @param {File} file - The image file to classify.
 * @returns {Promise<{prediction: string, confidence: number, cached: boolean}>}
 */
export async function predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await apiClient.post('/api/v1/predict', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    } catch (error) {
        if (error.response) {
            switch (error.response.status) {
                case 413:
                    throw new Error('Image is too large. Please upload a file under 5MB.');
                case 415:
                    throw new Error('Unsupported file type. Please upload a JPEG or PNG.');
                case 429:
                    throw new Error('Too many requests. Please wait a moment before trying again.');
                case 503:
                    throw new Error('The deep learning model is still warming up. Try again in a few seconds.');
                default:
                    throw new Error('An unexpected server error occurred.');
            }
        }
        throw new Error('Network error. Ensure the FastAPI server is running.');
    }
}
