export default function Problem() {
  return (
    <section id="problem" className="bg-white py-20 px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-3xl font-bold text-gray-900">The Problem</h2>
        <p className="mt-4 text-gray-700 leading-relaxed">
          NASA&apos;s satellite precipitation product provides global daily
          estimates, but at a coarse 0.1° resolution (~10&nbsp;km per pixel).
          This resolution is too coarse to capture the fine-scale rainfall
          gradients driven by Hawai&#8216;i&apos;s complex topography — a single
          pixel can span everything from a coastal rain shadow to a windward
          summit receiving over 10 meters of rain per year.
        </p>
        <p className="mt-4 text-gray-700 leading-relaxed">
          The goal of this project is to use a convolutional neural network to
          downscale the satellite data from 10&nbsp;km to{' '}
          <strong>250&nbsp;m resolution</strong>, using high-resolution
          topographic data (elevation, slope, aspect) and GOES-17 cloud cover as
          auxiliary inputs. Ground-truth supervision comes from ~165 rain gauge
          stations operated by the Hawai&#8216;i Climate Data Portal (HCDP).
        </p>
        <div className="mt-8 overflow-x-auto">
          <table className="min-w-full text-sm border border-gray-200">
            <thead className="bg-gray-100 text-left">
              <tr>
                <th className="px-4 py-2 font-semibold">Input</th>
                <th className="px-4 py-2 font-semibold">Source</th>
                <th className="px-4 py-2 font-semibold">Resolution</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Daily precipitation',  'NASA IMERG Early Run V07B', '~10 km'],
                ['Binary cloud mask',    'GOES-17 BCM',               '~2 km'],
                ['Elevation / Slope / Aspect', 'SRTM DEM',            '30 m'],
              ].map(([label, source, res]) => (
                <tr key={label} className="border-t border-gray-200">
                  <td className="px-4 py-2">{label}</td>
                  <td className="px-4 py-2 text-gray-500">{source}</td>
                  <td className="px-4 py-2">{res}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
