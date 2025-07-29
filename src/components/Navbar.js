import Link from 'next/link';

export default function Navbar() {
  return (
    <nav style={{ padding: '10px', borderBottom: '1px solid #ccc' }}>
      <Link href="/" style={{ marginRight: '15px' }}>Upload</Link>
      <Link href="/chat">Chatbot</Link>
    </nav>
  );
}
