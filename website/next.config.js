/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath: '/precipitation-downsampling',
  assetPrefix: '/precipitation-downsampling/',
  images: { unoptimized: true },
};

module.exports = nextConfig;
