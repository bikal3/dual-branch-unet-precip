import { render, screen } from '@testing-library/react';
import Problem from '@/components/Problem';

describe('Problem', () => {
  it('mentions IMERG and resolution', () => {
    render(<Problem />);
    // IMERG appears in at least one visible element
    const imergEls = screen.getAllByText(/IMERG/);
    expect(imergEls.length).toBeGreaterThan(0);
    expect(screen.getByText(/250\s*m/i)).toBeInTheDocument();
  });

  it('renders the section heading', () => {
    render(<Problem />);
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(/the problem/i);
  });

  it('renders the data sources table', () => {
    render(<Problem />);
    expect(screen.getByRole('columnheader', { name: /source/i })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: /NASA IMERG/i })).toBeInTheDocument();
  });

  it('has the correct section id', () => {
    const { container } = render(<Problem />);
    expect(container.querySelector('#problem')).not.toBeNull();
  });
});
