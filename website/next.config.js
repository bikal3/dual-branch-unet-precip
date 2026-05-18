/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/precipitation-downsampling',
  assetPrefix: '/precipitation-downsampling/',
  images: { unoptimized: true },
};

module.exports = nextConfig;
