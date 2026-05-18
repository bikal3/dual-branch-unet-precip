import { render, screen } from '@testing-library/react';
import NavBar from '@/components/NavBar';

describe('NavBar', () => {
  it('renders all section links', () => {
    render(<NavBar />);
    expect(screen.getByRole('link', { name: /problem/i })).toHaveAttribute('href', '#problem');
    expect(screen.getByRole('link', { name: /approach/i })).toHaveAttribute('href', '#approach');
    expect(screen.getByRole('link', { name: /results/i })).toHaveAttribute('href', '#results');
    expect(screen.getByRole('link', { name: /demo/i })).toHaveAttribute('href', '#demo');
    expect(screen.getByRole('link', { name: /about/i })).toHaveAttribute('href', '#about');
  });

  it('renders the project title', () => {
    render(<NavBar />);
    expect(screen.getByText(/precipitation downsampling/i)).toBeInTheDocument();
  });
});
