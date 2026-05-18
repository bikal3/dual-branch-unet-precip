interface ResultsProps {
  basePath: string;
}

const METRICS = [
  { label: 'Best val loss',     value: '0.0361', note: 'epoch 24 / 50' },
  { label: 'Test loss',         value: '0.0332', note: 'Apr–Jun 2021'  },
  { label: 'Train/val MSE gap', value: '~0.006', note: 'mild overfit'  },
  { label: 'Total parameters',  value: '2.3 M',  note: ''              },
];

export default function Results({ basePath }: ResultsProps) {
  return (
    <section id="results" className="bg-white py-20 px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-3xl font-bold text-gray-900">Results</h2>

        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full text-sm border border-gray-200">
            <thead className="bg-gray-100 text-left">
              <tr>
                <th className="px-4 py-2 font-semibold">Metric</th>
                <th className="px-4 py-2 font-semibold">Value</th>
                <th className="px-4 py-2 font-semibold text-gray-500">Note</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map(({ label, value, note }) => (
                <tr key={label} className="border-t border-gray-200">
                  <td className="px-4 py-2">{label}</td>
                  <td className="px-4 py-2 font-mono font-semibold">{value}</td>
                  <td className="px-4 py-2 text-gray-500">{note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 className="mt-8 text-xl font-semibold text-gray-800">
          Training Loss Curve
        </h3>
        <img
          src={`${basePath}/loss_curve.png`}
          alt="Loss curve showing train and validation loss over 50 epochs"
          className="mt-3 rounded-md border border-gray-200 w-full max-w-xl"
        />
      </div>
    </section>
  );
}
