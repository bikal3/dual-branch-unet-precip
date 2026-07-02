export default function Hero() {
  return (
    <section
      id="hero"
      className="flex flex-col items-center justify-center bg-gray-950 py-28 px-6 text-center text-white"
    >
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
        Precipitation Downscaling
      </h1>
      <p className="mt-4 max-w-2xl text-lg text-gray-300">
        Using a Dual-Branch U-Net CNN to downscale NASA IMERG satellite
        precipitation from 10&nbsp;km to 250&nbsp;m resolution over the Big
        Island of Hawai&#700;i.
      </p>
      <p className="mt-3 text-sm text-gray-400">
        By{' '}
        <a
          href="https://bikal3.com.np/"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-gray-200 hover:text-white hover:underline transition-colors"
        >
          Bikal Shrestha
        </a>
      </p>
      <div className="mt-8 flex gap-4">
        <a
          href="https://github.com/bikal3/dual-branch-unet-precip"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-md bg-white px-5 py-2.5 text-sm font-semibold text-gray-900 hover:bg-gray-200 transition-colors"
        >
          GitHub
        </a>
      </div>
    </section>
  );
}
