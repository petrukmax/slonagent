import { html, render, Component } from '../lib.js';

let _instance = null;

class LightboxView extends Component {
    constructor(props) {
        super(props);
        this.state = { group: null, index: 0 };
        _instance = this;
    }

    _images() {
        return [...document.querySelectorAll(`img[data-lightbox="${this.state.group}"]`)].map(el => el.dataset.full || el.src);
    }

    open(el) {
        const img = el.tagName === 'IMG' ? el : el.querySelector('img[data-lightbox]');
        if (!img) return;
        const group = img.dataset.lightbox;
        const all = [...document.querySelectorAll(`img[data-lightbox="${group}"]`)];
        const index = all.indexOf(img);
        this.setState({ group, index: Math.max(0, index) });
    }

    close() {
        this.setState({ group: null });
    }

    componentDidMount() {
        window.addEventListener('keydown', e => {
            if (!this.state.group) return;
            if (e.key === 'ArrowLeft') this.setState(s => ({ index: Math.max(0, s.index - 1) }));
            else if (e.key === 'ArrowRight') {
                const len = this._images().length;
                this.setState(s => ({ index: Math.min(len - 1, s.index + 1) }));
            } else if (e.key === 'Escape') this.close();
        });
    }

    render() {
        const { group, index } = this.state;
        if (!group) return null;
        const images = this._images();
        if (!images.length) return null;
        const src = images[index] || images[0];
        const hasPrev = index > 0;
        const hasNext = index < images.length - 1;

        return html`
            <div class="lightbox" onClick=${() => this.close()}>
                ${hasPrev && html`<div class="lb-arrow lb-prev" onClick=${e => { e.stopPropagation(); this.setState({ index: index - 1 }); }}>\u2039</div>`}
                <img src=${src} onClick=${e => e.stopPropagation()} />
                ${hasNext && html`<div class="lb-arrow lb-next" onClick=${e => { e.stopPropagation(); this.setState({ index: index + 1 }); }}>\u203A</div>`}
                <div class="lb-counter">${index + 1} / ${images.length}</div>
            </div>
        `;
    }
}

// Mount singleton
const _container = document.createElement('div');
document.body.appendChild(_container);
render(html`<${LightboxView} />`, _container);

export const Lightbox = {
    open(img) { _instance?.open(img); },
    close() { _instance?.close(); },
};
