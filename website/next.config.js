/** @type {import('next').NextConfig} */
const base = process.env.NEXT_PUBLIC_BASE_PATH || '';

const nextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath: base,
  assetPrefix: base ? `${base}/` : '',
  images: { unoptimized: true },
};

module.exports = nextConfig;
