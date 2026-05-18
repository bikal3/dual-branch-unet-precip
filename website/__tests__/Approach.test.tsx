import { render, screen } from '@testing-library/react';
import Approach from '@/components/Approach';

describe('Approach', () => {
  it('renders the architecture heading', () => {
    render(<Approach />);
    expect(screen.getByText(/dual.branch u.?net/i)).toBeInTheDocument();
  });

  it('mentions both branches', () => {
    render(<Approach />);
    expect(screen.getByText(/branch 1/i)).toBeInTheDocument();
    expect(screen.getByText(/branch 2/i)).toBeInTheDocument();
  });

  it('has the correct section id', () => {
    const { container } = render(<Approach />);
    expect(container.querySelector('#approach')).not.toBeNull();
  });
});
