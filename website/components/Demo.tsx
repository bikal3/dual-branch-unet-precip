import dynamic from 'next/dynamic';
import { useState } from 'react';

const DemoMap = dynamic(() => import('./DemoMap'), { ssr: false });

const SAMPLE_DATES: { label: string; value: string }[] = [
  { label: 'Jan 15', value: '2020-01-15' },
  { label: 'Mar 10', value: '2020-03-10' },
  { label: 'Jul 20', value: '2020-07-20' },
];

interface DemoProps {
  basePath: string;
}

export default function Demo({ basePath }: DemoProps) {
  const [selectedDate, setSelectedDate] = useState(SAMPLE_DATES[0].value);
  const [opacity, setOpacity] = useState(1);

  return (
    <section id="demo" className="bg-gray-50 py-20 px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-3xl font-bold text-gray-900">Interactive Demo</h2>
        <p className="mt-2 text-gray-600 text-sm">
          Slide to blend between the raw IMERG input (10 km) and the
          Dual-Branch U-Net prediction (250 m). Pick a date to load a different
          day.
        </p>

        {/* Day picker */}
        <div className="mt-6 flex gap-2">
          {SAMPLE_DATES.map(({ label, value }) => (
            <button
              key={value}
              onClick={() => setSelectedDate(value)}
              className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
                selectedDate === value
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-100'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Opacity slider */}
        <div className="mt-4 flex items-center gap-4">
          <span className="text-xs text-gray-500 w-16 text-right">IMERG</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1"
          />
          <span className="text-xs text-gray-500 w-24">Prediction</span>
        </div>

        {/* Map */}
        <div className="mt-4 rounded-md overflow-hidden border border-gray-200">
          <DemoMap date={selectedDate} opacity={opacity} basePath={basePath} />
        </div>

        {/* Legend */}
        <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
          <div
            className="h-3 w-32 rounded"
            style={{
              background:
                'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)',
            }}
          />
          <span>0 mm/day → high (normalised)</span>
        </div>
      </div>
    </section>
  );
}
