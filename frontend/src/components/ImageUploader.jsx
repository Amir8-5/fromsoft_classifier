import { useState, useRef } from 'react';
import { Upload } from 'lucide-react';

const MAX_SIZE = 5 * 1024 * 1024; // 5 MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png'];

export default function ImageUploader({ onFileSelect }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  function validate(file) {
    if (!ALLOWED_TYPES.includes(file.type)) {
      alert('Unsupported file type. Please upload a JPEG or PNG.');
      return false;
    }
    if (file.size > MAX_SIZE) {
      alert('Image is too large. Please upload a file under 5 MB.');
      return false;
    }
    return true;
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && validate(file)) onFileSelect(file);
  }

  function handleChange(e) {
    const file = e.target.files[0];
    if (file && validate(file)) onFileSelect(file);
  }

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all
        ${isDragging
          ? 'border-bb-gold bg-bb-bg'
          : 'border-bb-border bg-bb-panel'
        }`}
    >
      <Upload className="mx-auto text-bb-gold" size={48} />
      <p className="text-bb-text mt-4 text-lg">
        Drag & Drop Screenshot<br />or<br />Click to Upload
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png"
        className="hidden"
        onChange={handleChange}
      />
    </div>
  );
}
