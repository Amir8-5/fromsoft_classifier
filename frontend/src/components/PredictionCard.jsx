export default function PredictionCard({ prediction, confidence }) {
  const pct = (confidence * 100).toFixed(1);

  return (
    <div className="mt-8 text-center flex flex-col items-center">
      <p className="text-bb-text text-xl mb-2">Prediction:</p>
      <h2 className="text-bb-gold text-5xl md:text-6xl tracking-wider capitalize font-bold mb-6">
        {prediction}
      </h2>
      <div className="w-full max-w-md bg-bb-border h-6 rounded overflow-hidden">
        <div
          className="bg-bb-gold h-full transition-all duration-1000 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-bb-text mt-3">Confidence: {pct}%</p>
    </div>
  );
}
