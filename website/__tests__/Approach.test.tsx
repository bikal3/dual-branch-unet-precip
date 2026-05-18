import { render, screen } from '@testing-library/react';
import Approach from '@/components/Approach';

describe('Approach', () => {
  it('renders the architecture heading', () => {
    render(<Approach />);
    expect(screen.getByText(/dual.branch u.?net/i)).toBeInTheDocument();
  });

  it('shows Branch 1 and Branch 2 labels', () => {
    render(<Approach />);
    // The labelled <span> elements
    const spans = screen.getAllByText(/branch [12]/i);
    expect(spans.length).toBeGreaterThanOrEqual(2);
  });

  it('renders the architecture diagram in a pre block', () => {
    render(<Approach />);
    const pre = screen.getByTestId('arch-diagram');
    expect(pre).toBeInTheDocument();
    expect(pre.tagName).toBe('PRE');
    expect(pre).toHaveTextContent(/Branch 1/);
  });

  it('has the correct section id', () => {
    const { container } = render(<Approach />);
    expect(container.querySelector('#approach')).not.toBeNull();
  });
});
