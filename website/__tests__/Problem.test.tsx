import { render, screen } from '@testing-library/react';
import Problem from '@/components/Problem';

describe('Problem', () => {
  it('mentions IMERG and resolution', () => {
    render(<Problem />);
    expect(screen.getByText(/IMERG/)).toBeInTheDocument();
    expect(screen.getByText(/250\s*m/i)).toBeInTheDocument();
  });

  it('has the correct section id', () => {
    const { container } = render(<Problem />);
    expect(container.querySelector('#problem')).not.toBeNull();
  });
});
