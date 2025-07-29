import { NextRequest, NextResponse } from 'next/server';

const FLASK_BACKEND_URL = 'http://localhost:8000/upload';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('files') as File | null;
    const product_id = formData.get('product_id') || 'default_product_id';

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const fileBuffer = await file.arrayBuffer();
    const form = new FormData();
    form.append('files', new Blob([fileBuffer]), file.name);
    form.append('product_id', product_id as string);

    const response = await fetch(FLASK_BACKEND_URL, {
      method: 'POST',
      body: form as any, // Type assertion needed for Next.js
    });

    const result = await response.json();
    return NextResponse.json(result, { status: response.status });
  } catch (err: any) {
    console.error('Upload error:', err);
    return NextResponse.json(
      { error: err.message || 'Failed to process upload' },
      { status: 500 }
    );
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};
