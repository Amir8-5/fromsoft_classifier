import { useState } from 'react';
import { predictImage } from './api/client';
import ImageUploader from './components/ImageUploader';
import PredictionCard from './components/PredictionCard';
import LoadingOverlay from './components/LoadingOverlay';

export default function App() {
  const [appState, setAppState] = useState('idle');       // idle | predicting | success | error
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  async function handleImageUpload(file) {
    setAppState('predicting');
    setErrorMessage(null);

    try {
      const data = await predictImage(file);
      setPrediction(data.prediction);
      setConfidence(data.confidence);
      setAppState('success');
    } catch (error) {
      setErrorMessage(error.message);
      setAppState('error');
    }
  }

  return (
    <div className="min-h-screen bg-bb-bg py-16 px-4 font-serif flex flex-col items-center">
      <header className="mb-12 text-center">
        <h1 className="text-4xl md:text-5xl text-bb-gold tracking-widest mb-2">FromSoftware</h1>
        <p className="text-bb-text tracking-widest uppercase text-sm">Image Classifier</p>
      </header>

      <main className="w-full max-w-2xl">
        {(appState === 'idle' || appState === 'error') && (
          <>
            <ImageUploader onFileSelect={handleImageUpload} />
            {appState === 'error' && (
              <div className="mt-6 border border-red-900 bg-[#2a0808] text-red-200 p-4 rounded text-center">
                {errorMessage}
              </div>
            )}
          </>
        )}

        {appState === 'predicting' && <LoadingOverlay />}

        {appState === 'success' && (
          <>
            <PredictionCard prediction={prediction} confidence={confidence} />
            <button
              onClick={() => setAppState('idle')}
              className="mt-12 mx-auto block px-6 py-2 border border-bb-border text-bb-text hover:border-bb-gold hover:text-bb-gold transition-colors tracking-widest"
            >
              Classify Another
            </button>
          </>
        )}
      </main>
    </div>
  );
}
