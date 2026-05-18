/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath: '/dual-branch-unet-precip',
  assetPrefix: '/dual-branch-unet-precip/',
  images: { unoptimized: true },
};

module.exports = nextConfig;
