import { render, screen } from '@testing-library/react';
import Results from '@/components/Results';

describe('Results', () => {
  it('shows the best val loss', () => {
    render(<Results basePath="" />);
    expect(screen.getByText('0.0361')).toBeInTheDocument();
  });

  it('shows the test loss', () => {
    render(<Results basePath="" />);
    expect(screen.getByText('0.0332')).toBeInTheDocument();
  });

  it('renders the loss curve image', () => {
    render(<Results basePath="/precipitation-downsampling" />);
    const img = screen.getByRole('img', { name: /loss curve/i });
    expect(img).toHaveAttribute('src', '/precipitation-downsampling/loss_curve.png');
  });

  it('has the correct section id', () => {
    const { container } = render(<Results basePath="" />);
    expect(container.querySelector('#results')).not.toBeNull();
  });
});
