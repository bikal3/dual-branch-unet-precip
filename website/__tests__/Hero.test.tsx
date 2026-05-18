import { render, screen } from '@testing-library/react';
import Hero from '@/components/Hero';

describe('Hero', () => {
  it('renders the project title', () => {
    render(<Hero />);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
      /precipitation downsampling/i
    );
  });

  it('renders a GitHub link to the correct repo', () => {
    render(<Hero />);
    const link = screen.getByRole('link', { name: /github/i });
    expect(link).toHaveAttribute('href', 'https://github.com/bikal3/precipitation-downsampling');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });
});
