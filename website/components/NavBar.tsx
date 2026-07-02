const LINKS = [
  { label: 'Problem',  href: '#problem'  },
  { label: 'Approach', href: '#approach' },
  { label: 'Results',  href: '#results'  },
  { label: 'Demo',     href: '#demo'     },
  { label: 'About',    href: '#about'    },
];

export default function NavBar() {
  return (
    <nav className="sticky top-0 z-50 bg-gray-900 text-white shadow-md">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3">
        <span className="font-semibold text-sm tracking-wide">
          Precipitation Downscaling
        </span>
        <ul className="hidden sm:flex gap-6 text-sm">
          {LINKS.map(({ label, href }) => (
            <li key={href}>
              <a href={href} className="hover:text-blue-400 transition-colors">
                {label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
