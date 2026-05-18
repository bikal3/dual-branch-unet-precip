const TEAM: { name: string; role: string }[] = [
  { name: 'Bikal',     role: 'Data acquisition, Dual-Branch U-Net' },
  { name: 'Elisabeth', role: 'Reprojection, DataLoader, DA-Net'    },
  { name: 'Gabby',     role: 'Station rasterisation, Nested U-Net' },
];

export default function About() {
  return (
    <section id="about" className="bg-white py-20 px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-3xl font-bold text-gray-900">About</h2>
        <p className="mt-4 text-gray-700">
          This project was completed as part of the{' '}
          <strong>ADLEO course</strong>. The goal was to compare three CNN
          architectures — Dual-Branch U-Net, Nested U-Net, and DA-Net — for
          precipitation downscaling over the Big Island of Hawai&#8216;i.
        </p>

        <h3 className="mt-8 text-xl font-semibold text-gray-800">Team</h3>
        <ul className="mt-3 space-y-2">
          {TEAM.map(({ name, role }) => (
            <li key={name} className="flex gap-3 text-gray-700">
              <span className="font-semibold w-24 shrink-0">{name}</span>
              <span className="text-gray-500">{role}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
