import { render, screen } from '@testing-library/react';
import Approach from '@/components/Approach';

describe('Approach', () => {
  it('renders the architecture heading', () => {
    render(<Approach />);
    expect(screen.getByText(/dual.branch u.?net/i)).toBeInTheDocument();
  });

  it('mentions both branches', () => {
    render(<Approach />);
    expect(screen.getAllByText(/branch 1/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/branch 2/i).length).toBeGreaterThan(0);
  });

  it('has the correct section id', () => {
    const { container } = render(<Approach />);
    expect(container.querySelector('#approach')).not.toBeNull();
  });
});
