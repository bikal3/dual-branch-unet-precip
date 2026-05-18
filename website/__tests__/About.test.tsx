import { render, screen } from '@testing-library/react';
import About from '@/components/About';

describe('About', () => {
  it('renders special thanks with GitHub links', () => {
    render(<About />);
    expect(screen.getByRole('link', { name: /elisabeth tappert/i })).toHaveAttribute('href', 'https://github.com/ETappert');
    expect(screen.getByRole('link', { name: /gabriela de leon/i })).toHaveAttribute('href', 'https://github.com/gabdele');
  });

  it('describes the project', () => {
    render(<About />);
    expect(screen.getByText(/IMERG/)).toBeInTheDocument();
    expect(screen.getByText(/250\s*m/i)).toBeInTheDocument();
  });

  it('renders the section heading', () => {
    render(<About />);
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(/about/i);
  });

  it('has the correct section id', () => {
    const { container } = render(<About />);
    expect(container.querySelector('#about')).not.toBeNull();
  });
});
