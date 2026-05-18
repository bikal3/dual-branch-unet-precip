export default function About() {
  return (
    <section id="about" className="bg-white py-20 px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-3xl font-bold text-gray-900">About</h2>
        <p className="mt-4 text-gray-700 leading-relaxed">
          This project uses a Dual-Branch U-Net convolutional neural network to
          downscale NASA IMERG precipitation data from 10&nbsp;km to 250&nbsp;m
          resolution over the Big Island of Hawai&#700;i. The model fuses
          satellite precipitation and cloud cover with high-resolution
          topographic data to produce fine-scale daily rainfall fields,
          supervised by ~165 ground-truth rain gauge stations from the
          Hawai&#700;i Climate Data Portal.
        </p>

        <h3 className="mt-8 text-xl font-semibold text-gray-800">Special Thanks</h3>
        <p className="mt-3 text-gray-700 leading-relaxed">
          Special thanks to{' '}
          <a href="https://github.com/ETappert" target="_blank" rel="noopener noreferrer" className="font-semibold text-blue-600 hover:underline">Elisabeth Tappert</a>
          {' '}and{' '}
          <a href="https://github.com/gabdele" target="_blank" rel="noopener noreferrer" className="font-semibold text-blue-600 hover:underline">Gabriela de Leon</a>
          {' '}for their support throughout this project — helping with data reprojection,
          building the input DataLoader pipeline, and rasterising rain gauge station data into
          training targets.
        </p>
      </div>
    </section>
  );
}
