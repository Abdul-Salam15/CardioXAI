(function () {
  function build() {
    if (!window.React || !window.ReactDOM || !window.Recharts) return false;
    var React = window.React;
    var R = window.Recharts;
    var h = React.createElement;

    function Chart(props) {
      var data = props.data || [];
      return h(R.ResponsiveContainer, { width: '100%', height: '100%' },
        h(R.BarChart, { data: data, layout: 'vertical', margin: { top: 6, right: 52, left: 6, bottom: 6 } },
          h(R.CartesianGrid, { horizontal: false, stroke: '#f1f5f9' }),
          h(R.XAxis, {
            type: 'number', domain: [-0.2, 0.4],
            tickFormatter: function (v) { return v.toFixed(2); },
            tick: { fontSize: 11, fill: '#9ca3af' }, axisLine: false, tickLine: false
          }),
          h(R.YAxis, {
            type: 'category', dataKey: 'name', width: 96,
            tick: { fontSize: 12, fill: '#374151', fontWeight: 600 },
            axisLine: false, tickLine: false
          }),
          h(R.ReferenceLine, { x: 0, stroke: '#cbd5e1', strokeWidth: 1.5 }),
          h(R.Tooltip, {
            cursor: { fill: 'rgba(0,0,0,0.04)' },
            formatter: function (v) { return [(v > 0 ? '+' : '') + Number(v).toFixed(2), 'SHAP value']; },
            contentStyle: { borderRadius: 10, border: '1px solid #e5e7eb', fontSize: 12, fontWeight: 600 }
          }),
          h(R.Bar, { dataKey: 'value', radius: [0, 4, 4, 0], barSize: 18, isAnimationActive: true, animationDuration: 900 },
            data.map(function (d, i) {
              return h(R.Cell, { key: i, fill: d.value >= 0 ? '#B71C1C' : '#1A73E8' });
            }),
            h(R.LabelList, {
              dataKey: 'value', position: 'right',
              formatter: function (v) { return (v > 0 ? '+' : '') + Number(v).toFixed(2); },
              style: { fontSize: 11, fontWeight: 700, fill: '#6b7280' }
            })
          )
        )
      );
    }

    class ShapChart extends HTMLElement {
      static get observedAttributes() { return ['data']; }
      connectedCallback() { this._mount(); }
      attributeChangedCallback() { this._mount(); }
      disconnectedCallback() { if (this._root) { try { this._root.unmount(); } catch (e) {} this._root = null; } }
      _mount() {
        var data = [];
        try { data = JSON.parse(this.getAttribute('data') || '[]'); } catch (e) { data = []; }
        this.style.display = 'block';
        this.style.width = '100%';
        this.style.height = '100%';
        if (!this._root) this._root = window.ReactDOM.createRoot(this);
        this._root.render(h(Chart, { data: data }));
      }
    }
    if (!customElements.get('shap-chart')) customElements.define('shap-chart', ShapChart);
    return true;
  }

  if (!build()) {
    var tries = 0;
    var iv = setInterval(function () { if (build() || tries++ > 60) clearInterval(iv); }, 100);
  }
})();
