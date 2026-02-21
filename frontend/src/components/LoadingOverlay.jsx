export default function LoadingOverlay() {
  return (
    <div className="flex flex-col items-center justify-center mt-12 animate-pulse">
      <div className="w-12 h-12 border-t-2 border-bb-gold rounded-full animate-spin mb-4" />
      <p className="text-bb-gold text-xl tracking-widest">Consulting the flames...</p>
    </div>
  );
}
