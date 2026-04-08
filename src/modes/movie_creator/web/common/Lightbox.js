import { html, render, Component } from '../lib.js';

let _instance = null;

class LightboxView extends Component {
    constructor(props) {
        super(props);
        this.state = { group: null, index: 0 };
        _instance = this;
    }

    _items() {
        return [...document.querySelectorAll(`img[data-lightbox="${this.state.group}"]`)].map(el => ({
            src: el.dataset.full || el.src,
            isVideo: !!el.dataset.video,
        }));
    }

    open(el) {
        const media = el.tagName === 'IMG' ? el : el.querySelector('img[data-lightbox]');
        if (!media?.dataset?.lightbox) return;
        const group = media.dataset.lightbox;
        const all = [...document.querySelectorAll(`img[data-lightbox="${group}"]`)];
        const index = all.indexOf(media);
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
                const len = this._items().length;
                this.setState(s => ({ index: Math.min(len - 1, s.index + 1) }));
            } else if (e.key === 'Escape') this.close();
        });
    }

    render() {
        const { group, index } = this.state;
        if (!group) return null;
        const items = this._items();
        if (!items.length) return null;
        const item = items[index] || items[0];
        const hasPrev = index > 0;
        const hasNext = index < items.length - 1;

        return html`
            <div class="lightbox" onClick=${() => this.close()}>
                ${hasPrev && html`<div class="lb-arrow lb-prev" onClick=${e => { e.stopPropagation(); this.setState({ index: index - 1 }); }}>\u2039</div>`}
                ${item.isVideo
                    ? html`<video src=${item.src} controls autoplay onClick=${e => e.stopPropagation()} />`
                    : html`<img src=${item.src} onClick=${e => e.stopPropagation()} />`}
                ${hasNext && html`<div class="lb-arrow lb-next" onClick=${e => { e.stopPropagation(); this.setState({ index: index + 1 }); }}>\u203A</div>`}
                <div class="lb-counter">${index + 1} / ${items.length}</div>
            </div>
        `;
    }
}

// Mount singleton
const _container = document.createElement('div');
document.body.appendChild(_container);
render(html`<${LightboxView} />`, _container);

export const Lightbox = {
    open(el) { _instance?.open(el); },
    close() { _instance?.close(); },
};
