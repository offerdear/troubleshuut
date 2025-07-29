'use client'
import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("files", file);
    formData.append("product_id", "default_product_id");

    setStatus("Uploading...");

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const json = await res.json();
      if (res.ok) {
        setStatus(`✅ Uploaded. Chunks: ${json.chunks_stored || json.chunks || 'N/A'}`);
      } else {
        setStatus(`❌ Error: ${json.error || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Upload failed:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setStatus(`❌ Upload failed: ${errorMessage}`);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Upload a Document</h1>
      <input type="file" accept=".pdf,.docx" onChange={e => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleUpload} disabled={!file}>Upload</button>
      {status && <p>{status}</p>}
    </div>
  );
}
