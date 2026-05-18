import { render, screen } from '@testing-library/react';
import About from '@/components/About';

describe('About', () => {
  it('renders all team members', () => {
    render(<About />);
    expect(screen.getByText(/bikal/i)).toBeInTheDocument();
    expect(screen.getByText(/elisabeth/i)).toBeInTheDocument();
    expect(screen.getByText(/gabby/i)).toBeInTheDocument();
  });

  it('mentions the course', () => {
    render(<About />);
    expect(screen.getByText(/adleo/i)).toBeInTheDocument();
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
