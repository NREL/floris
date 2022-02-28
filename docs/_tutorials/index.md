---
layout: default
title: Overview
nav_order: 1
---


<html>
  <head><meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
  <title>Overview</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
  
  
  <style type="text/css">
      pre { line-height: 125%; }
  td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
  span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
  td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
  span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
  .highlight .hll { background-color: var(--jp-cell-editor-active-background) }
  .highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
  .highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
  .highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
  .highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
  .highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
  .highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
  .highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
  .highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
  .highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
  .highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
  .highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
  .highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
  .highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
  .highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
  .highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
  .highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
  .highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
  .highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
  .highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
  .highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
  .highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
  .highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
  .highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
  .highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
  .highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
  .highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
  .highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
  .highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
  .highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
  .highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
  .highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
  .highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
  .highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
  .highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
  .highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
  .highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
  .highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
  .highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
  .highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
  .highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
  .highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
    </style>
  
  
  
  <!-- <style type="text/css">
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*
   * Mozilla scrollbar styling
   */
  
  /* use standard opaque scrollbars for most nodes */
  [data-jp-theme-scrollbars='true'] {
    scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
      var(--jp-scrollbar-background-color);
  }
  
  /* for code nodes, use a transparent style of scrollbar. These selectors
   * will match lower in the tree, and so will override the above */
  [data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
  [data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
    scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  }
  
  /* tiny scrollbar */
  
  .jp-scrollbar-tiny {
    scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
    scrollbar-width: thin;
  }
  
  /*
   * Webkit scrollbar styling
   */
  
  /* use standard opaque scrollbars for most nodes */
  
  [data-jp-theme-scrollbars='true'] ::-webkit-scrollbar,
  [data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-corner {
    background: var(--jp-scrollbar-background-color);
  }
  
  [data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-thumb {
    background: rgb(var(--jp-scrollbar-thumb-color));
    border: var(--jp-scrollbar-thumb-margin) solid transparent;
    background-clip: content-box;
    border-radius: var(--jp-scrollbar-thumb-radius);
  }
  
  [data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:horizontal {
    border-left: var(--jp-scrollbar-endpad) solid
      var(--jp-scrollbar-background-color);
    border-right: var(--jp-scrollbar-endpad) solid
      var(--jp-scrollbar-background-color);
  }
  
  [data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:vertical {
    border-top: var(--jp-scrollbar-endpad) solid
      var(--jp-scrollbar-background-color);
    border-bottom: var(--jp-scrollbar-endpad) solid
      var(--jp-scrollbar-background-color);
  }
  
  /* for code nodes, use a transparent style of scrollbar */
  
  [data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar::-webkit-scrollbar,
  [data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar::-webkit-scrollbar,
  [data-jp-theme-scrollbars='true']
    .CodeMirror-hscrollbar::-webkit-scrollbar-corner,
  [data-jp-theme-scrollbars='true']
    .CodeMirror-vscrollbar::-webkit-scrollbar-corner {
    background-color: transparent;
  }
  
  [data-jp-theme-scrollbars='true']
    .CodeMirror-hscrollbar::-webkit-scrollbar-thumb,
  [data-jp-theme-scrollbars='true']
    .CodeMirror-vscrollbar::-webkit-scrollbar-thumb {
    background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
    border: var(--jp-scrollbar-thumb-margin) solid transparent;
    background-clip: content-box;
    border-radius: var(--jp-scrollbar-thumb-radius);
  }
  
  [data-jp-theme-scrollbars='true']
    .CodeMirror-hscrollbar::-webkit-scrollbar-track:horizontal {
    border-left: var(--jp-scrollbar-endpad) solid transparent;
    border-right: var(--jp-scrollbar-endpad) solid transparent;
  }
  
  [data-jp-theme-scrollbars='true']
    .CodeMirror-vscrollbar::-webkit-scrollbar-track:vertical {
    border-top: var(--jp-scrollbar-endpad) solid transparent;
    border-bottom: var(--jp-scrollbar-endpad) solid transparent;
  }
  
  /* tiny scrollbar */
  
  .jp-scrollbar-tiny::-webkit-scrollbar,
  .jp-scrollbar-tiny::-webkit-scrollbar-corner {
    background-color: transparent;
    height: 4px;
    width: 4px;
  }
  
  .jp-scrollbar-tiny::-webkit-scrollbar-thumb {
    background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
  }
  
  .jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
    border-left: 0px solid transparent;
    border-right: 0px solid transparent;
  }
  
  .jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
    border-top: 0px solid transparent;
    border-bottom: 0px solid transparent;
  }
  
  /*
   * Phosphor
   */
  
  .lm-ScrollBar[data-orientation='horizontal'] {
    min-height: 16px;
    max-height: 16px;
    min-width: 45px;
    border-top: 1px solid #a0a0a0;
  }
  
  .lm-ScrollBar[data-orientation='vertical'] {
    min-width: 16px;
    max-width: 16px;
    min-height: 45px;
    border-left: 1px solid #a0a0a0;
  }
  
  .lm-ScrollBar-button {
    background-color: #f0f0f0;
    background-position: center center;
    min-height: 15px;
    max-height: 15px;
    min-width: 15px;
    max-width: 15px;
  }
  
  .lm-ScrollBar-button:hover {
    background-color: #dadada;
  }
  
  .lm-ScrollBar-button.lm-mod-active {
    background-color: #cdcdcd;
  }
  
  .lm-ScrollBar-track {
    background: #f0f0f0;
  }
  
  .lm-ScrollBar-thumb {
    background: #cdcdcd;
  }
  
  .lm-ScrollBar-thumb:hover {
    background: #bababa;
  }
  
  .lm-ScrollBar-thumb.lm-mod-active {
    background: #a0a0a0;
  }
  
  .lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
    height: 100%;
    min-width: 15px;
    border-left: 1px solid #a0a0a0;
    border-right: 1px solid #a0a0a0;
  }
  
  .lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
    width: 100%;
    min-height: 15px;
    border-top: 1px solid #a0a0a0;
    border-bottom: 1px solid #a0a0a0;
  }
  
  .lm-ScrollBar[data-orientation='horizontal']
    .lm-ScrollBar-button[data-action='decrement'] {
    background-image: var(--jp-icon-caret-left);
    background-size: 17px;
  }
  
  .lm-ScrollBar[data-orientation='horizontal']
    .lm-ScrollBar-button[data-action='increment'] {
    background-image: var(--jp-icon-caret-right);
    background-size: 17px;
  }
  
  .lm-ScrollBar[data-orientation='vertical']
    .lm-ScrollBar-button[data-action='decrement'] {
    background-image: var(--jp-icon-caret-up);
    background-size: 17px;
  }
  
  .lm-ScrollBar[data-orientation='vertical']
    .lm-ScrollBar-button[data-action='increment'] {
    background-image: var(--jp-icon-caret-down);
    background-size: 17px;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-Widget, /* </DEPRECATED> */
  .lm-Widget {
    box-sizing: border-box;
    position: relative;
    overflow: hidden;
    cursor: default;
  }
  
  
  /* <DEPRECATED> */ .p-Widget.p-mod-hidden, /* </DEPRECATED> */
  .lm-Widget.lm-mod-hidden {
    display: none !important;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-CommandPalette, /* </DEPRECATED> */
  .lm-CommandPalette {
    display: flex;
    flex-direction: column;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-search, /* </DEPRECATED> */
  .lm-CommandPalette-search {
    flex: 0 0 auto;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-content, /* </DEPRECATED> */
  .lm-CommandPalette-content {
    flex: 1 1 auto;
    margin: 0;
    padding: 0;
    min-height: 0;
    overflow: auto;
    list-style-type: none;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-header, /* </DEPRECATED> */
  .lm-CommandPalette-header {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-item, /* </DEPRECATED> */
  .lm-CommandPalette-item {
    display: flex;
    flex-direction: row;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-itemIcon, /* </DEPRECATED> */
  .lm-CommandPalette-itemIcon {
    flex: 0 0 auto;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-itemContent, /* </DEPRECATED> */
  .lm-CommandPalette-itemContent {
    flex: 1 1 auto;
    overflow: hidden;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-itemShortcut, /* </DEPRECATED> */
  .lm-CommandPalette-itemShortcut {
    flex: 0 0 auto;
  }
  
  
  /* <DEPRECATED> */ .p-CommandPalette-itemLabel, /* </DEPRECATED> */
  .lm-CommandPalette-itemLabel {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  
  .lm-close-icon {
    border:1px solid transparent;
    background-color: transparent;
    position: absolute;
    z-index:1;
    right:3%;
    top: 0;
    bottom: 0;
    margin: auto;
    padding: 7px 0;
    display: none;
    vertical-align: middle;
    outline: 0;
    cursor: pointer;
  }
  .lm-close-icon:after {
    content: "X";
    display: block;
    width: 15px;
    height: 15px;
    text-align: center;
    color:#000;
    font-weight: normal;
    font-size: 12px;
    cursor: pointer;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-DockPanel, /* </DEPRECATED> */
  .lm-DockPanel {
    z-index: 0;
  }
  
  
  /* <DEPRECATED> */ .p-DockPanel-widget, /* </DEPRECATED> */
  .lm-DockPanel-widget {
    z-index: 0;
  }
  
  
  /* <DEPRECATED> */ .p-DockPanel-tabBar, /* </DEPRECATED> */
  .lm-DockPanel-tabBar {
    z-index: 1;
  }
  
  
  /* <DEPRECATED> */ .p-DockPanel-handle, /* </DEPRECATED> */
  .lm-DockPanel-handle {
    z-index: 2;
  }
  
  
  /* <DEPRECATED> */ .p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
  .lm-DockPanel-handle.lm-mod-hidden {
    display: none !important;
  }
  
  
  /* <DEPRECATED> */ .p-DockPanel-handle:after, /* </DEPRECATED> */
  .lm-DockPanel-handle:after {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    content: '';
  }
  
  
  /* <DEPRECATED> */
  .p-DockPanel-handle[data-orientation='horizontal'],
  /* </DEPRECATED> */
  .lm-DockPanel-handle[data-orientation='horizontal'] {
    cursor: ew-resize;
  }
  
  
  /* <DEPRECATED> */
  .p-DockPanel-handle[data-orientation='vertical'],
  /* </DEPRECATED> */
  .lm-DockPanel-handle[data-orientation='vertical'] {
    cursor: ns-resize;
  }
  
  
  /* <DEPRECATED> */
  .p-DockPanel-handle[data-orientation='horizontal']:after,
  /* </DEPRECATED> */
  .lm-DockPanel-handle[data-orientation='horizontal']:after {
    left: 50%;
    min-width: 8px;
    transform: translateX(-50%);
  }
  
  
  /* <DEPRECATED> */
  .p-DockPanel-handle[data-orientation='vertical']:after,
  /* </DEPRECATED> */
  .lm-DockPanel-handle[data-orientation='vertical']:after {
    top: 50%;
    min-height: 8px;
    transform: translateY(-50%);
  }
  
  
  /* <DEPRECATED> */ .p-DockPanel-overlay, /* </DEPRECATED> */
  .lm-DockPanel-overlay {
    z-index: 3;
    box-sizing: border-box;
    pointer-events: none;
  }
  
  
  /* <DEPRECATED> */ .p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
  .lm-DockPanel-overlay.lm-mod-hidden {
    display: none !important;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-Menu, /* </DEPRECATED> */
  .lm-Menu {
    z-index: 10000;
    position: absolute;
    white-space: nowrap;
    overflow-x: hidden;
    overflow-y: auto;
    outline: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  
  /* <DEPRECATED> */ .p-Menu-content, /* </DEPRECATED> */
  .lm-Menu-content {
    margin: 0;
    padding: 0;
    display: table;
    list-style-type: none;
  }
  
  
  /* <DEPRECATED> */ .p-Menu-item, /* </DEPRECATED> */
  .lm-Menu-item {
    display: table-row;
  }
  
  
  /* <DEPRECATED> */
  .p-Menu-item.p-mod-hidden,
  .p-Menu-item.p-mod-collapsed,
  /* </DEPRECATED> */
  .lm-Menu-item.lm-mod-hidden,
  .lm-Menu-item.lm-mod-collapsed {
    display: none !important;
  }
  
  
  /* <DEPRECATED> */
  .p-Menu-itemIcon,
  .p-Menu-itemSubmenuIcon,
  /* </DEPRECATED> */
  .lm-Menu-itemIcon,
  .lm-Menu-itemSubmenuIcon {
    display: table-cell;
    text-align: center;
  }
  
  
  /* <DEPRECATED> */ .p-Menu-itemLabel, /* </DEPRECATED> */
  .lm-Menu-itemLabel {
    display: table-cell;
    text-align: left;
  }
  
  
  /* <DEPRECATED> */ .p-Menu-itemShortcut, /* </DEPRECATED> */
  .lm-Menu-itemShortcut {
    display: table-cell;
    text-align: right;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-MenuBar, /* </DEPRECATED> */
  .lm-MenuBar {
    outline: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  
  /* <DEPRECATED> */ .p-MenuBar-content, /* </DEPRECATED> */
  .lm-MenuBar-content {
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: row;
    list-style-type: none;
  }
  
  
  /* <DEPRECATED> */ .p--MenuBar-item, /* </DEPRECATED> */
  .lm-MenuBar-item {
    box-sizing: border-box;
  }
  
  
  /* <DEPRECATED> */
  .p-MenuBar-itemIcon,
  .p-MenuBar-itemLabel,
  /* </DEPRECATED> */
  .lm-MenuBar-itemIcon,
  .lm-MenuBar-itemLabel {
    display: inline-block;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-ScrollBar, /* </DEPRECATED> */
  .lm-ScrollBar {
    display: flex;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  
  /* <DEPRECATED> */
  .p-ScrollBar[data-orientation='horizontal'],
  /* </DEPRECATED> */
  .lm-ScrollBar[data-orientation='horizontal'] {
    flex-direction: row;
  }
  
  
  /* <DEPRECATED> */
  .p-ScrollBar[data-orientation='vertical'],
  /* </DEPRECATED> */
  .lm-ScrollBar[data-orientation='vertical'] {
    flex-direction: column;
  }
  
  
  /* <DEPRECATED> */ .p-ScrollBar-button, /* </DEPRECATED> */
  .lm-ScrollBar-button {
    box-sizing: border-box;
    flex: 0 0 auto;
  }
  
  
  /* <DEPRECATED> */ .p-ScrollBar-track, /* </DEPRECATED> */
  .lm-ScrollBar-track {
    box-sizing: border-box;
    position: relative;
    overflow: hidden;
    flex: 1 1 auto;
  }
  
  
  /* <DEPRECATED> */ .p-ScrollBar-thumb, /* </DEPRECATED> */
  .lm-ScrollBar-thumb {
    box-sizing: border-box;
    position: absolute;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-SplitPanel-child, /* </DEPRECATED> */
  .lm-SplitPanel-child {
    z-index: 0;
  }
  
  
  /* <DEPRECATED> */ .p-SplitPanel-handle, /* </DEPRECATED> */
  .lm-SplitPanel-handle {
    z-index: 1;
  }
  
  
  /* <DEPRECATED> */ .p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
  .lm-SplitPanel-handle.lm-mod-hidden {
    display: none !important;
  }
  
  
  /* <DEPRECATED> */ .p-SplitPanel-handle:after, /* </DEPRECATED> */
  .lm-SplitPanel-handle:after {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    content: '';
  }
  
  
  /* <DEPRECATED> */
  .p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
  /* </DEPRECATED> */
  .lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
    cursor: ew-resize;
  }
  
  
  /* <DEPRECATED> */
  .p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
  /* </DEPRECATED> */
  .lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
    cursor: ns-resize;
  }
  
  
  /* <DEPRECATED> */
  .p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
  /* </DEPRECATED> */
  .lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
    left: 50%;
    min-width: 8px;
    transform: translateX(-50%);
  }
  
  
  /* <DEPRECATED> */
  .p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
  /* </DEPRECATED> */
  .lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
    top: 50%;
    min-height: 8px;
    transform: translateY(-50%);
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-TabBar, /* </DEPRECATED> */
  .lm-TabBar {
    display: flex;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  
  /* <DEPRECATED> */ .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
  .lm-TabBar[data-orientation='horizontal'] {
    flex-direction: row;
  }
  
  
  /* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
  .lm-TabBar[data-orientation='vertical'] {
    flex-direction: column;
  }
  
  
  /* <DEPRECATED> */ .p-TabBar-content, /* </DEPRECATED> */
  .lm-TabBar-content {
    margin: 0;
    padding: 0;
    display: flex;
    flex: 1 1 auto;
    list-style-type: none;
  }
  
  
  /* <DEPRECATED> */
  .p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
  /* </DEPRECATED> */
  .lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
    flex-direction: row;
  }
  
  
  /* <DEPRECATED> */
  .p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
  /* </DEPRECATED> */
  .lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
    flex-direction: column;
  }
  
  
  /* <DEPRECATED> */ .p-TabBar-tab, /* </DEPRECATED> */
  .lm-TabBar-tab {
    display: flex;
    flex-direction: row;
    box-sizing: border-box;
    overflow: hidden;
  }
  
  
  /* <DEPRECATED> */
  .p-TabBar-tabIcon,
  .p-TabBar-tabCloseIcon,
  /* </DEPRECATED> */
  .lm-TabBar-tabIcon,
  .lm-TabBar-tabCloseIcon {
    flex: 0 0 auto;
  }
  
  
  /* <DEPRECATED> */ .p-TabBar-tabLabel, /* </DEPRECATED> */
  .lm-TabBar-tabLabel {
    flex: 1 1 auto;
    overflow: hidden;
    white-space: nowrap;
  }
  
  
  .lm-TabBar-tabInput {
    user-select: all;
    width: 100%;
    box-sizing : border-box;
  }
  
  
  /* <DEPRECATED> */ .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
  .lm-TabBar-tab.lm-mod-hidden {
    display: none !important;
  }
  
  
  /* <DEPRECATED> */ .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
  .lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
    position: relative;
  }
  
  
  /* <DEPRECATED> */
  .p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
  /* </DEPRECATED> */
  .lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
    left: 0;
    transition: left 150ms ease;
  }
  
  
  /* <DEPRECATED> */
  .p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
  /* </DEPRECATED> */
  .lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
    top: 0;
    transition: top 150ms ease;
  }
  
  
  /* <DEPRECATED> */
  .p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging,
  /* </DEPRECATED> */
  .lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
    transition: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ .p-TabPanel-tabBar, /* </DEPRECATED> */
  .lm-TabPanel-tabBar {
    z-index: 1;
  }
  
  
  /* <DEPRECATED> */ .p-TabPanel-stackedPanel, /* </DEPRECATED> */
  .lm-TabPanel-stackedPanel {
    z-index: 0;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  @charset "UTF-8";
  html{
    -webkit-box-sizing:border-box;
            box-sizing:border-box; }
  
  *,
  *::before,
  *::after{
    -webkit-box-sizing:inherit;
            box-sizing:inherit; }
  
  body{
    font-size:14px;
    font-weight:400;
    letter-spacing:0;
    line-height:1.28581;
    text-transform:none;
    color:#182026;
    font-family:-apple-system, "BlinkMacSystemFont", "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Open Sans", "Helvetica Neue", "Icons16", sans-serif; }
  
  p{
    margin-bottom:10px;
    margin-top:0; }
  
  small{
    font-size:12px; }
  
  strong{
    font-weight:600; }
  
  ::-moz-selection{
    background:rgba(125, 188, 255, 0.6); }
  
  ::selection{
    background:rgba(125, 188, 255, 0.6); }
  .bp3-heading{
    color:#182026;
    font-weight:600;
    margin:0 0 10px;
    padding:0; }
    .bp3-dark .bp3-heading{
      color:#f5f8fa; }
  
  h1.bp3-heading, .bp3-running-text h1{
    font-size:36px;
    line-height:40px; }
  
  h2.bp3-heading, .bp3-running-text h2{
    font-size:28px;
    line-height:32px; }
  
  h3.bp3-heading, .bp3-running-text h3{
    font-size:22px;
    line-height:25px; }
  
  h4.bp3-heading, .bp3-running-text h4{
    font-size:18px;
    line-height:21px; }
  
  h5.bp3-heading, .bp3-running-text h5{
    font-size:16px;
    line-height:19px; }
  
  h6.bp3-heading, .bp3-running-text h6{
    font-size:14px;
    line-height:16px; }
  .bp3-ui-text{
    font-size:14px;
    font-weight:400;
    letter-spacing:0;
    line-height:1.28581;
    text-transform:none; }
  
  .bp3-monospace-text{
    font-family:monospace;
    text-transform:none; }
  
  .bp3-text-muted{
    color:#5c7080; }
    .bp3-dark .bp3-text-muted{
      color:#a7b6c2; }
  
  .bp3-text-disabled{
    color:rgba(92, 112, 128, 0.6); }
    .bp3-dark .bp3-text-disabled{
      color:rgba(167, 182, 194, 0.6); }
  
  .bp3-text-overflow-ellipsis{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal; }
  .bp3-running-text{
    font-size:14px;
    line-height:1.5; }
    .bp3-running-text h1{
      color:#182026;
      font-weight:600;
      margin-bottom:20px;
      margin-top:40px; }
      .bp3-dark .bp3-running-text h1{
        color:#f5f8fa; }
    .bp3-running-text h2{
      color:#182026;
      font-weight:600;
      margin-bottom:20px;
      margin-top:40px; }
      .bp3-dark .bp3-running-text h2{
        color:#f5f8fa; }
    .bp3-running-text h3{
      color:#182026;
      font-weight:600;
      margin-bottom:20px;
      margin-top:40px; }
      .bp3-dark .bp3-running-text h3{
        color:#f5f8fa; }
    .bp3-running-text h4{
      color:#182026;
      font-weight:600;
      margin-bottom:20px;
      margin-top:40px; }
      .bp3-dark .bp3-running-text h4{
        color:#f5f8fa; }
    .bp3-running-text h5{
      color:#182026;
      font-weight:600;
      margin-bottom:20px;
      margin-top:40px; }
      .bp3-dark .bp3-running-text h5{
        color:#f5f8fa; }
    .bp3-running-text h6{
      color:#182026;
      font-weight:600;
      margin-bottom:20px;
      margin-top:40px; }
      .bp3-dark .bp3-running-text h6{
        color:#f5f8fa; }
    .bp3-running-text hr{
      border:none;
      border-bottom:1px solid rgba(16, 22, 26, 0.15);
      margin:20px 0; }
      .bp3-dark .bp3-running-text hr{
        border-color:rgba(255, 255, 255, 0.15); }
    .bp3-running-text p{
      margin:0 0 10px;
      padding:0; }
  
  .bp3-text-large{
    font-size:16px; }
  
  .bp3-text-small{
    font-size:12px; }
  a{
    color:#106ba3;
    text-decoration:none; }
    a:hover{
      color:#106ba3;
      cursor:pointer;
      text-decoration:underline; }
    a .bp3-icon, a .bp3-icon-standard, a .bp3-icon-large{
      color:inherit; }
    a code,
    .bp3-dark a code{
      color:inherit; }
    .bp3-dark a,
    .bp3-dark a:hover{
      color:#48aff0; }
      .bp3-dark a .bp3-icon, .bp3-dark a .bp3-icon-standard, .bp3-dark a .bp3-icon-large,
      .bp3-dark a:hover .bp3-icon,
      .bp3-dark a:hover .bp3-icon-standard,
      .bp3-dark a:hover .bp3-icon-large{
        color:inherit; }
  .bp3-running-text code, .bp3-code{
    font-family:monospace;
    text-transform:none;
    background:rgba(255, 255, 255, 0.7);
    border-radius:3px;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
    color:#5c7080;
    font-size:smaller;
    padding:2px 5px; }
    .bp3-dark .bp3-running-text code, .bp3-running-text .bp3-dark code, .bp3-dark .bp3-code{
      background:rgba(16, 22, 26, 0.3);
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#a7b6c2; }
    .bp3-running-text a > code, a > .bp3-code{
      color:#137cbd; }
      .bp3-dark .bp3-running-text a > code, .bp3-running-text .bp3-dark a > code, .bp3-dark a > .bp3-code{
        color:inherit; }
  
  .bp3-running-text pre, .bp3-code-block{
    font-family:monospace;
    text-transform:none;
    background:rgba(255, 255, 255, 0.7);
    border-radius:3px;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
    color:#182026;
    display:block;
    font-size:13px;
    line-height:1.4;
    margin:10px 0;
    padding:13px 15px 12px;
    word-break:break-all;
    word-wrap:break-word; }
    .bp3-dark .bp3-running-text pre, .bp3-running-text .bp3-dark pre, .bp3-dark .bp3-code-block{
      background:rgba(16, 22, 26, 0.3);
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
    .bp3-running-text pre > code, .bp3-code-block > code{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit;
      font-size:inherit;
      padding:0; }
  
  .bp3-running-text kbd, .bp3-key{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    background:#ffffff;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
    color:#5c7080;
    display:-webkit-inline-box;
    display:-ms-inline-flexbox;
    display:inline-flex;
    font-family:inherit;
    font-size:12px;
    height:24px;
    -webkit-box-pack:center;
        -ms-flex-pack:center;
            justify-content:center;
    line-height:24px;
    min-width:24px;
    padding:3px 6px;
    vertical-align:middle; }
    .bp3-running-text kbd .bp3-icon, .bp3-key .bp3-icon, .bp3-running-text kbd .bp3-icon-standard, .bp3-key .bp3-icon-standard, .bp3-running-text kbd .bp3-icon-large, .bp3-key .bp3-icon-large{
      margin-right:5px; }
    .bp3-dark .bp3-running-text kbd, .bp3-running-text .bp3-dark kbd, .bp3-dark .bp3-key{
      background:#394b59;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
      color:#a7b6c2; }
  .bp3-running-text blockquote, .bp3-blockquote{
    border-left:solid 4px rgba(167, 182, 194, 0.5);
    margin:0 0 10px;
    padding:0 20px; }
    .bp3-dark .bp3-running-text blockquote, .bp3-running-text .bp3-dark blockquote, .bp3-dark .bp3-blockquote{
      border-color:rgba(115, 134, 148, 0.5); }
  .bp3-running-text ul,
  .bp3-running-text ol, .bp3-list{
    margin:10px 0;
    padding-left:30px; }
    .bp3-running-text ul li:not(:last-child), .bp3-running-text ol li:not(:last-child), .bp3-list li:not(:last-child){
      margin-bottom:5px; }
    .bp3-running-text ul ol, .bp3-running-text ol ol, .bp3-list ol,
    .bp3-running-text ul ul,
    .bp3-running-text ol ul,
    .bp3-list ul{
      margin-top:5px; }
  
  .bp3-list-unstyled{
    list-style:none;
    margin:0;
    padding:0; }
    .bp3-list-unstyled li{
      padding:0; }
  .bp3-rtl{
    text-align:right; }
  
  .bp3-dark{
    color:#f5f8fa; }
  
  :focus{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:2px;
    -moz-outline-radius:6px; }
  
  .bp3-focus-disabled :focus{
    outline:none !important; }
    .bp3-focus-disabled :focus ~ .bp3-control-indicator{
      outline:none !important; }
  
  .bp3-alert{
    max-width:400px;
    padding:20px; }
  
  .bp3-alert-body{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }
    .bp3-alert-body .bp3-icon{
      font-size:40px;
      margin-right:20px;
      margin-top:0; }
  
  .bp3-alert-contents{
    word-break:break-word; }
  
  .bp3-alert-footer{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:reverse;
        -ms-flex-direction:row-reverse;
            flex-direction:row-reverse;
    margin-top:10px; }
    .bp3-alert-footer .bp3-button{
      margin-left:10px; }
  .bp3-breadcrumbs{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    cursor:default;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -ms-flex-wrap:wrap;
        flex-wrap:wrap;
    height:30px;
    list-style:none;
    margin:0;
    padding:0; }
    .bp3-breadcrumbs > li{
      -webkit-box-align:center;
          -ms-flex-align:center;
              align-items:center;
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex; }
      .bp3-breadcrumbs > li::after{
        background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M10.71 7.29l-4-4a1.003 1.003 0 00-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 001.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z' fill='%235C7080'/%3e%3c/svg%3e");
        content:"";
        display:block;
        height:16px;
        margin:0 5px;
        width:16px; }
      .bp3-breadcrumbs > li:last-of-type::after{
        display:none; }
  
  .bp3-breadcrumb,
  .bp3-breadcrumb-current,
  .bp3-breadcrumbs-collapsed{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-inline-box;
    display:-ms-inline-flexbox;
    display:inline-flex;
    font-size:16px; }
  
  .bp3-breadcrumb,
  .bp3-breadcrumbs-collapsed{
    color:#5c7080; }
  
  .bp3-breadcrumb:hover{
    text-decoration:none; }
  
  .bp3-breadcrumb.bp3-disabled{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  
  .bp3-breadcrumb .bp3-icon{
    margin-right:5px; }
  
  .bp3-breadcrumb-current{
    color:inherit;
    font-weight:600; }
    .bp3-breadcrumb-current .bp3-input{
      font-size:inherit;
      font-weight:inherit;
      vertical-align:baseline; }
  
  .bp3-breadcrumbs-collapsed{
    background:#ced9e0;
    border:none;
    border-radius:3px;
    cursor:pointer;
    margin-right:2px;
    padding:1px 5px;
    vertical-align:text-bottom; }
    .bp3-breadcrumbs-collapsed::before{
      background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cg fill='%235C7080'%3e%3ccircle cx='2' cy='8.03' r='2'/%3e%3ccircle cx='14' cy='8.03' r='2'/%3e%3ccircle cx='8' cy='8.03' r='2'/%3e%3c/g%3e%3c/svg%3e") center no-repeat;
      content:"";
      display:block;
      height:16px;
      width:16px; }
    .bp3-breadcrumbs-collapsed:hover{
      background:#bfccd6;
      color:#182026;
      text-decoration:none; }
  
  .bp3-dark .bp3-breadcrumb,
  .bp3-dark .bp3-breadcrumbs-collapsed{
    color:#a7b6c2; }
  
  .bp3-dark .bp3-breadcrumbs > li::after{
    color:#a7b6c2; }
  
  .bp3-dark .bp3-breadcrumb.bp3-disabled{
    color:rgba(167, 182, 194, 0.6); }
  
  .bp3-dark .bp3-breadcrumb-current{
    color:#f5f8fa; }
  
  .bp3-dark .bp3-breadcrumbs-collapsed{
    background:rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-breadcrumbs-collapsed:hover{
      background:rgba(16, 22, 26, 0.6);
      color:#f5f8fa; }
  .bp3-button{
    display:-webkit-inline-box;
    display:-ms-inline-flexbox;
    display:inline-flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    border:none;
    border-radius:3px;
    cursor:pointer;
    font-size:14px;
    -webkit-box-pack:center;
        -ms-flex-pack:center;
            justify-content:center;
    padding:5px 10px;
    text-align:left;
    vertical-align:middle;
    min-height:30px;
    min-width:30px; }
    .bp3-button > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-button > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-button::before,
    .bp3-button > *{
      margin-right:7px; }
    .bp3-button:empty::before,
    .bp3-button > :last-child{
      margin-right:0; }
    .bp3-button:empty{
      padding:0 !important; }
    .bp3-button:disabled, .bp3-button.bp3-disabled{
      cursor:not-allowed; }
    .bp3-button.bp3-fill{
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex;
      width:100%; }
    .bp3-button.bp3-align-right,
    .bp3-align-right .bp3-button{
      text-align:right; }
    .bp3-button.bp3-align-left,
    .bp3-align-left .bp3-button{
      text-align:left; }
    .bp3-button:not([class*="bp3-intent-"]){
      background-color:#f5f8fa;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      color:#182026; }
      .bp3-button:not([class*="bp3-intent-"]):hover{
        background-clip:padding-box;
        background-color:#ebf1f5;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
      .bp3-button:not([class*="bp3-intent-"]):active, .bp3-button:not([class*="bp3-intent-"]).bp3-active{
        background-color:#d8e1e8;
        background-image:none;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed;
        outline:none; }
        .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active:hover, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
    .bp3-button.bp3-intent-primary{
      background-color:#137cbd;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      color:#ffffff; }
      .bp3-button.bp3-intent-primary:hover, .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
        color:#ffffff; }
      .bp3-button.bp3-intent-primary:hover{
        background-color:#106ba3;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
        background-color:#0e5a8a;
        background-image:none;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-primary:disabled, .bp3-button.bp3-intent-primary.bp3-disabled{
        background-color:rgba(19, 124, 189, 0.5);
        background-image:none;
        border-color:transparent;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(255, 255, 255, 0.6); }
    .bp3-button.bp3-intent-success{
      background-color:#0f9960;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      color:#ffffff; }
      .bp3-button.bp3-intent-success:hover, .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
        color:#ffffff; }
      .bp3-button.bp3-intent-success:hover{
        background-color:#0d8050;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
        background-color:#0a6640;
        background-image:none;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-success:disabled, .bp3-button.bp3-intent-success.bp3-disabled{
        background-color:rgba(15, 153, 96, 0.5);
        background-image:none;
        border-color:transparent;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(255, 255, 255, 0.6); }
    .bp3-button.bp3-intent-warning{
      background-color:#d9822b;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      color:#ffffff; }
      .bp3-button.bp3-intent-warning:hover, .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
        color:#ffffff; }
      .bp3-button.bp3-intent-warning:hover{
        background-color:#bf7326;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
        background-color:#a66321;
        background-image:none;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-warning:disabled, .bp3-button.bp3-intent-warning.bp3-disabled{
        background-color:rgba(217, 130, 43, 0.5);
        background-image:none;
        border-color:transparent;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(255, 255, 255, 0.6); }
    .bp3-button.bp3-intent-danger{
      background-color:#db3737;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      color:#ffffff; }
      .bp3-button.bp3-intent-danger:hover, .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
        color:#ffffff; }
      .bp3-button.bp3-intent-danger:hover{
        background-color:#c23030;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
        background-color:#a82a2a;
        background-image:none;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-button.bp3-intent-danger:disabled, .bp3-button.bp3-intent-danger.bp3-disabled{
        background-color:rgba(219, 55, 55, 0.5);
        background-image:none;
        border-color:transparent;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(255, 255, 255, 0.6); }
    .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
      stroke:#ffffff; }
    .bp3-button.bp3-large,
    .bp3-large .bp3-button{
      min-height:40px;
      min-width:40px;
      font-size:16px;
      padding:5px 15px; }
      .bp3-button.bp3-large::before,
      .bp3-button.bp3-large > *,
      .bp3-large .bp3-button::before,
      .bp3-large .bp3-button > *{
        margin-right:10px; }
      .bp3-button.bp3-large:empty::before,
      .bp3-button.bp3-large > :last-child,
      .bp3-large .bp3-button:empty::before,
      .bp3-large .bp3-button > :last-child{
        margin-right:0; }
    .bp3-button.bp3-small,
    .bp3-small .bp3-button{
      min-height:24px;
      min-width:24px;
      padding:0 7px; }
    .bp3-button.bp3-loading{
      position:relative; }
      .bp3-button.bp3-loading[class*="bp3-icon-"]::before{
        visibility:hidden; }
      .bp3-button.bp3-loading .bp3-button-spinner{
        margin:0;
        position:absolute; }
      .bp3-button.bp3-loading > :not(.bp3-button-spinner){
        visibility:hidden; }
    .bp3-button[class*="bp3-icon-"]::before{
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      color:#5c7080; }
    .bp3-button .bp3-icon, .bp3-button .bp3-icon-standard, .bp3-button .bp3-icon-large{
      color:#5c7080; }
      .bp3-button .bp3-icon.bp3-align-right, .bp3-button .bp3-icon-standard.bp3-align-right, .bp3-button .bp3-icon-large.bp3-align-right{
        margin-left:7px; }
    .bp3-button .bp3-icon:first-child:last-child,
    .bp3-button .bp3-spinner + .bp3-icon:last-child{
      margin:0 -7px; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]){
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover, .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover{
        background-color:#30404d;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
        background-color:#202b33;
        background-image:none;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"])[class*="bp3-icon-"]::before{
        color:#a7b6c2; }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-large{
        color:#a7b6c2; }
    .bp3-dark .bp3-button[class*="bp3-intent-"]{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-button[class*="bp3-intent-"]:hover{
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-button[class*="bp3-intent-"]:active, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-active{
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-button[class*="bp3-intent-"]:disabled, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-disabled{
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(255, 255, 255, 0.3); }
      .bp3-dark .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
        stroke:#8a9ba8; }
    .bp3-button:disabled::before,
    .bp3-button:disabled .bp3-icon, .bp3-button:disabled .bp3-icon-standard, .bp3-button:disabled .bp3-icon-large, .bp3-button.bp3-disabled::before,
    .bp3-button.bp3-disabled .bp3-icon, .bp3-button.bp3-disabled .bp3-icon-standard, .bp3-button.bp3-disabled .bp3-icon-large, .bp3-button[class*="bp3-intent-"]::before,
    .bp3-button[class*="bp3-intent-"] .bp3-icon, .bp3-button[class*="bp3-intent-"] .bp3-icon-standard, .bp3-button[class*="bp3-intent-"] .bp3-icon-large{
      color:inherit !important; }
    .bp3-button.bp3-minimal{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none; }
      .bp3-button.bp3-minimal:hover{
        background:rgba(167, 182, 194, 0.3);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#182026;
        text-decoration:none; }
      .bp3-button.bp3-minimal:active, .bp3-button.bp3-minimal.bp3-active{
        background:rgba(115, 134, 148, 0.3);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#182026; }
      .bp3-button.bp3-minimal:disabled, .bp3-button.bp3-minimal:disabled:hover, .bp3-button.bp3-minimal.bp3-disabled, .bp3-button.bp3-minimal.bp3-disabled:hover{
        background:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed; }
        .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
          background:rgba(115, 134, 148, 0.3); }
      .bp3-dark .bp3-button.bp3-minimal{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:inherit; }
        .bp3-dark .bp3-button.bp3-minimal:hover, .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none; }
        .bp3-dark .bp3-button.bp3-minimal:hover{
          background:rgba(138, 155, 168, 0.15); }
        .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
          background:rgba(138, 155, 168, 0.3);
          color:#f5f8fa; }
        .bp3-dark .bp3-button.bp3-minimal:disabled, .bp3-dark .bp3-button.bp3-minimal:disabled:hover, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover{
          background:none;
          color:rgba(167, 182, 194, 0.6);
          cursor:not-allowed; }
          .bp3-dark .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
            background:rgba(138, 155, 168, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-primary{
        color:#106ba3; }
        .bp3-button.bp3-minimal.bp3-intent-primary:hover, .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#106ba3; }
        .bp3-button.bp3-minimal.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.15);
          color:#106ba3; }
        .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#106ba3; }
        .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(16, 107, 163, 0.5); }
          .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
        .bp3-button.bp3-minimal.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
          stroke:#106ba3; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary{
          color:#48aff0; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:hover{
            background:rgba(19, 124, 189, 0.2);
            color:#48aff0; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
            background:rgba(19, 124, 189, 0.3);
            color:#48aff0; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
            background:none;
            color:rgba(72, 175, 240, 0.5); }
            .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
              background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-success{
        color:#0d8050; }
        .bp3-button.bp3-minimal.bp3-intent-success:hover, .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#0d8050; }
        .bp3-button.bp3-minimal.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.15);
          color:#0d8050; }
        .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#0d8050; }
        .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(13, 128, 80, 0.5); }
          .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
        .bp3-button.bp3-minimal.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
          stroke:#0d8050; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success{
          color:#3dcc91; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:hover{
            background:rgba(15, 153, 96, 0.2);
            color:#3dcc91; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
            background:rgba(15, 153, 96, 0.3);
            color:#3dcc91; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
            background:none;
            color:rgba(61, 204, 145, 0.5); }
            .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
              background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-warning{
        color:#bf7326; }
        .bp3-button.bp3-minimal.bp3-intent-warning:hover, .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#bf7326; }
        .bp3-button.bp3-minimal.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.15);
          color:#bf7326; }
        .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#bf7326; }
        .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(191, 115, 38, 0.5); }
          .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
        .bp3-button.bp3-minimal.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
          stroke:#bf7326; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning{
          color:#ffb366; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:hover{
            background:rgba(217, 130, 43, 0.2);
            color:#ffb366; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
            background:rgba(217, 130, 43, 0.3);
            color:#ffb366; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
            background:none;
            color:rgba(255, 179, 102, 0.5); }
            .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
              background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-danger{
        color:#c23030; }
        .bp3-button.bp3-minimal.bp3-intent-danger:hover, .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#c23030; }
        .bp3-button.bp3-minimal.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.15);
          color:#c23030; }
        .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#c23030; }
        .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(194, 48, 48, 0.5); }
          .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
        .bp3-button.bp3-minimal.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
          stroke:#c23030; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger{
          color:#ff7373; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:hover{
            background:rgba(219, 55, 55, 0.2);
            color:#ff7373; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
            background:rgba(219, 55, 55, 0.3);
            color:#ff7373; }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
            background:none;
            color:rgba(255, 115, 115, 0.5); }
            .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
              background:rgba(219, 55, 55, 0.3); }
    .bp3-button.bp3-outlined{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      border:1px solid rgba(24, 32, 38, 0.2);
      -webkit-box-sizing:border-box;
              box-sizing:border-box; }
      .bp3-button.bp3-outlined:hover{
        background:rgba(167, 182, 194, 0.3);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#182026;
        text-decoration:none; }
      .bp3-button.bp3-outlined:active, .bp3-button.bp3-outlined.bp3-active{
        background:rgba(115, 134, 148, 0.3);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#182026; }
      .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined.bp3-disabled:hover{
        background:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed; }
        .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
          background:rgba(115, 134, 148, 0.3); }
      .bp3-dark .bp3-button.bp3-outlined{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:inherit; }
        .bp3-dark .bp3-button.bp3-outlined:hover, .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none; }
        .bp3-dark .bp3-button.bp3-outlined:hover{
          background:rgba(138, 155, 168, 0.15); }
        .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
          background:rgba(138, 155, 168, 0.3);
          color:#f5f8fa; }
        .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
          background:none;
          color:rgba(167, 182, 194, 0.6);
          cursor:not-allowed; }
          .bp3-dark .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
            background:rgba(138, 155, 168, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-primary{
        color:#106ba3; }
        .bp3-button.bp3-outlined.bp3-intent-primary:hover, .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#106ba3; }
        .bp3-button.bp3-outlined.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.15);
          color:#106ba3; }
        .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#106ba3; }
        .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(16, 107, 163, 0.5); }
          .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
        .bp3-button.bp3-outlined.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
          stroke:#106ba3; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
          color:#48aff0; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:hover{
            background:rgba(19, 124, 189, 0.2);
            color:#48aff0; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
            background:rgba(19, 124, 189, 0.3);
            color:#48aff0; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
            background:none;
            color:rgba(72, 175, 240, 0.5); }
            .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
              background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-success{
        color:#0d8050; }
        .bp3-button.bp3-outlined.bp3-intent-success:hover, .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#0d8050; }
        .bp3-button.bp3-outlined.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.15);
          color:#0d8050; }
        .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#0d8050; }
        .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(13, 128, 80, 0.5); }
          .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
        .bp3-button.bp3-outlined.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
          stroke:#0d8050; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
          color:#3dcc91; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:hover{
            background:rgba(15, 153, 96, 0.2);
            color:#3dcc91; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
            background:rgba(15, 153, 96, 0.3);
            color:#3dcc91; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
            background:none;
            color:rgba(61, 204, 145, 0.5); }
            .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
              background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-warning{
        color:#bf7326; }
        .bp3-button.bp3-outlined.bp3-intent-warning:hover, .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#bf7326; }
        .bp3-button.bp3-outlined.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.15);
          color:#bf7326; }
        .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#bf7326; }
        .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(191, 115, 38, 0.5); }
          .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
        .bp3-button.bp3-outlined.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
          stroke:#bf7326; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
          color:#ffb366; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:hover{
            background:rgba(217, 130, 43, 0.2);
            color:#ffb366; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
            background:rgba(217, 130, 43, 0.3);
            color:#ffb366; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
            background:none;
            color:rgba(255, 179, 102, 0.5); }
            .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
              background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-danger{
        color:#c23030; }
        .bp3-button.bp3-outlined.bp3-intent-danger:hover, .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#c23030; }
        .bp3-button.bp3-outlined.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.15);
          color:#c23030; }
        .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#c23030; }
        .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(194, 48, 48, 0.5); }
          .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
        .bp3-button.bp3-outlined.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
          stroke:#c23030; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
          color:#ff7373; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:hover{
            background:rgba(219, 55, 55, 0.2);
            color:#ff7373; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
            background:rgba(219, 55, 55, 0.3);
            color:#ff7373; }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
            background:none;
            color:rgba(255, 115, 115, 0.5); }
            .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
              background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled:hover{
        border-color:rgba(92, 112, 128, 0.1); }
      .bp3-dark .bp3-button.bp3-outlined{
        border-color:rgba(255, 255, 255, 0.4); }
        .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
          border-color:rgba(255, 255, 255, 0.2); }
      .bp3-button.bp3-outlined.bp3-intent-primary{
        border-color:rgba(16, 107, 163, 0.6); }
        .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          border-color:rgba(16, 107, 163, 0.2); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
          border-color:rgba(72, 175, 240, 0.6); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
            border-color:rgba(72, 175, 240, 0.2); }
      .bp3-button.bp3-outlined.bp3-intent-success{
        border-color:rgba(13, 128, 80, 0.6); }
        .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          border-color:rgba(13, 128, 80, 0.2); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
          border-color:rgba(61, 204, 145, 0.6); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
            border-color:rgba(61, 204, 145, 0.2); }
      .bp3-button.bp3-outlined.bp3-intent-warning{
        border-color:rgba(191, 115, 38, 0.6); }
        .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          border-color:rgba(191, 115, 38, 0.2); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
          border-color:rgba(255, 179, 102, 0.6); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
            border-color:rgba(255, 179, 102, 0.2); }
      .bp3-button.bp3-outlined.bp3-intent-danger{
        border-color:rgba(194, 48, 48, 0.6); }
        .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          border-color:rgba(194, 48, 48, 0.2); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
          border-color:rgba(255, 115, 115, 0.6); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
            border-color:rgba(255, 115, 115, 0.2); }
  
  a.bp3-button{
    text-align:center;
    text-decoration:none;
    -webkit-transition:none;
    transition:none; }
    a.bp3-button, a.bp3-button:hover, a.bp3-button:active{
      color:#182026; }
    a.bp3-button.bp3-disabled{
      color:rgba(92, 112, 128, 0.6); }
  
  .bp3-button-text{
    -webkit-box-flex:0;
        -ms-flex:0 1 auto;
            flex:0 1 auto; }
  
  .bp3-button.bp3-align-left .bp3-button-text, .bp3-button.bp3-align-right .bp3-button-text,
  .bp3-button-group.bp3-align-left .bp3-button-text,
  .bp3-button-group.bp3-align-right .bp3-button-text{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group{
    display:-webkit-inline-box;
    display:-ms-inline-flexbox;
    display:inline-flex; }
    .bp3-button-group .bp3-button{
      -webkit-box-flex:0;
          -ms-flex:0 0 auto;
              flex:0 0 auto;
      position:relative;
      z-index:4; }
      .bp3-button-group .bp3-button:focus{
        z-index:5; }
      .bp3-button-group .bp3-button:hover{
        z-index:6; }
      .bp3-button-group .bp3-button:active, .bp3-button-group .bp3-button.bp3-active{
        z-index:7; }
      .bp3-button-group .bp3-button:disabled, .bp3-button-group .bp3-button.bp3-disabled{
        z-index:3; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]{
        z-index:9; }
        .bp3-button-group .bp3-button[class*="bp3-intent-"]:focus{
          z-index:10; }
        .bp3-button-group .bp3-button[class*="bp3-intent-"]:hover{
          z-index:11; }
        .bp3-button-group .bp3-button[class*="bp3-intent-"]:active, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-active{
          z-index:12; }
        .bp3-button-group .bp3-button[class*="bp3-intent-"]:disabled, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-disabled{
          z-index:8; }
    .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:first-child) .bp3-button,
    .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:first-child){
      border-bottom-left-radius:0;
      border-top-left-radius:0; }
    .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
      border-bottom-right-radius:0;
      border-top-right-radius:0;
      margin-right:-1px; }
    .bp3-button-group.bp3-minimal .bp3-button{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none; }
      .bp3-button-group.bp3-minimal .bp3-button:hover{
        background:rgba(167, 182, 194, 0.3);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#182026;
        text-decoration:none; }
      .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:rgba(115, 134, 148, 0.3);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#182026; }
      .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
        background:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed; }
        .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
          background:rgba(115, 134, 148, 0.3); }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:inherit; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover{
          background:rgba(138, 155, 168, 0.15); }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
          background:rgba(138, 155, 168, 0.3);
          color:#f5f8fa; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
          background:none;
          color:rgba(167, 182, 194, 0.6);
          cursor:not-allowed; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
            background:rgba(138, 155, 168, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
        color:#106ba3; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#106ba3; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.15);
          color:#106ba3; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#106ba3; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(16, 107, 163, 0.5); }
          .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
          stroke:#106ba3; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
          color:#48aff0; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
            background:rgba(19, 124, 189, 0.2);
            color:#48aff0; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
            background:rgba(19, 124, 189, 0.3);
            color:#48aff0; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
            background:none;
            color:rgba(72, 175, 240, 0.5); }
            .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
              background:rgba(19, 124, 189, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
        color:#0d8050; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#0d8050; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.15);
          color:#0d8050; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#0d8050; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(13, 128, 80, 0.5); }
          .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
          stroke:#0d8050; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
          color:#3dcc91; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
            background:rgba(15, 153, 96, 0.2);
            color:#3dcc91; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
            background:rgba(15, 153, 96, 0.3);
            color:#3dcc91; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
            background:none;
            color:rgba(61, 204, 145, 0.5); }
            .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
              background:rgba(15, 153, 96, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
        color:#bf7326; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#bf7326; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.15);
          color:#bf7326; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#bf7326; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(191, 115, 38, 0.5); }
          .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
          stroke:#bf7326; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
          color:#ffb366; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
            background:rgba(217, 130, 43, 0.2);
            color:#ffb366; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
            background:rgba(217, 130, 43, 0.3);
            color:#ffb366; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
            background:none;
            color:rgba(255, 179, 102, 0.5); }
            .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
              background:rgba(217, 130, 43, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
        color:#c23030; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:#c23030; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.15);
          color:#c23030; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#c23030; }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(194, 48, 48, 0.5); }
          .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
          stroke:#c23030; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
          color:#ff7373; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
            background:rgba(219, 55, 55, 0.2);
            color:#ff7373; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
            background:rgba(219, 55, 55, 0.3);
            color:#ff7373; }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
            background:none;
            color:rgba(255, 115, 115, 0.5); }
            .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
              background:rgba(219, 55, 55, 0.3); }
    .bp3-button-group .bp3-popover-wrapper,
    .bp3-button-group .bp3-popover-target{
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex;
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto; }
    .bp3-button-group.bp3-fill{
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex;
      width:100%; }
    .bp3-button-group .bp3-button.bp3-fill,
    .bp3-button-group.bp3-fill .bp3-button:not(.bp3-fixed){
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto; }
    .bp3-button-group.bp3-vertical{
      -webkit-box-align:stretch;
          -ms-flex-align:stretch;
              align-items:stretch;
      -webkit-box-orient:vertical;
      -webkit-box-direction:normal;
          -ms-flex-direction:column;
              flex-direction:column;
      vertical-align:top; }
      .bp3-button-group.bp3-vertical.bp3-fill{
        height:100%;
        width:unset; }
      .bp3-button-group.bp3-vertical .bp3-button{
        margin-right:0 !important;
        width:100%; }
      .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:first-child .bp3-button,
      .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:first-child{
        border-radius:3px 3px 0 0; }
      .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:last-child .bp3-button,
      .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:last-child{
        border-radius:0 0 3px 3px; }
      .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
      .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:not(:last-child){
        margin-bottom:-1px; }
    .bp3-button-group.bp3-align-left .bp3-button{
      text-align:left; }
    .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
      margin-right:1px; }
    .bp3-dark .bp3-button-group.bp3-vertical > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-dark .bp3-button-group.bp3-vertical > .bp3-button:not(:last-child){
      margin-bottom:1px; }
  .bp3-callout{
    font-size:14px;
    line-height:1.5;
    background-color:rgba(138, 155, 168, 0.15);
    border-radius:3px;
    padding:10px 12px 9px;
    position:relative;
    width:100%; }
    .bp3-callout[class*="bp3-icon-"]{
      padding-left:40px; }
      .bp3-callout[class*="bp3-icon-"]::before{
        font-family:"Icons20", sans-serif;
        font-size:20px;
        font-style:normal;
        font-weight:400;
        line-height:1;
        -moz-osx-font-smoothing:grayscale;
        -webkit-font-smoothing:antialiased;
        color:#5c7080;
        left:10px;
        position:absolute;
        top:10px; }
    .bp3-callout.bp3-callout-icon{
      padding-left:40px; }
      .bp3-callout.bp3-callout-icon > .bp3-icon:first-child{
        color:#5c7080;
        left:10px;
        position:absolute;
        top:10px; }
    .bp3-callout .bp3-heading{
      line-height:20px;
      margin-bottom:5px;
      margin-top:0; }
      .bp3-callout .bp3-heading:last-child{
        margin-bottom:0; }
    .bp3-dark .bp3-callout{
      background-color:rgba(138, 155, 168, 0.2); }
      .bp3-dark .bp3-callout[class*="bp3-icon-"]::before{
        color:#a7b6c2; }
    .bp3-callout.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.15); }
      .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
      .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
      .bp3-callout.bp3-intent-primary .bp3-heading{
        color:#106ba3; }
      .bp3-dark .bp3-callout.bp3-intent-primary{
        background-color:rgba(19, 124, 189, 0.25); }
        .bp3-dark .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
        .bp3-dark .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
        .bp3-dark .bp3-callout.bp3-intent-primary .bp3-heading{
          color:#48aff0; }
    .bp3-callout.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.15); }
      .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
      .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
      .bp3-callout.bp3-intent-success .bp3-heading{
        color:#0d8050; }
      .bp3-dark .bp3-callout.bp3-intent-success{
        background-color:rgba(15, 153, 96, 0.25); }
        .bp3-dark .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
        .bp3-dark .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
        .bp3-dark .bp3-callout.bp3-intent-success .bp3-heading{
          color:#3dcc91; }
    .bp3-callout.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.15); }
      .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
      .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
      .bp3-callout.bp3-intent-warning .bp3-heading{
        color:#bf7326; }
      .bp3-dark .bp3-callout.bp3-intent-warning{
        background-color:rgba(217, 130, 43, 0.25); }
        .bp3-dark .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
        .bp3-dark .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
        .bp3-dark .bp3-callout.bp3-intent-warning .bp3-heading{
          color:#ffb366; }
    .bp3-callout.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.15); }
      .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
      .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
      .bp3-callout.bp3-intent-danger .bp3-heading{
        color:#c23030; }
      .bp3-dark .bp3-callout.bp3-intent-danger{
        background-color:rgba(219, 55, 55, 0.25); }
        .bp3-dark .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
        .bp3-dark .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
        .bp3-dark .bp3-callout.bp3-intent-danger .bp3-heading{
          color:#ff7373; }
    .bp3-running-text .bp3-callout{
      margin:20px 0; }
  .bp3-card{
    background-color:#ffffff;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
    padding:20px;
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-card.bp3-dark,
    .bp3-dark .bp3-card{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  
  .bp3-elevation-0{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
    .bp3-elevation-0.bp3-dark,
    .bp3-dark .bp3-elevation-0{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  
  .bp3-elevation-1{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-elevation-1.bp3-dark,
    .bp3-dark .bp3-elevation-1{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  
  .bp3-elevation-2{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2); }
    .bp3-elevation-2.bp3-dark,
    .bp3-dark .bp3-elevation-2{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4); }
  
  .bp3-elevation-3{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
    .bp3-elevation-3.bp3-dark,
    .bp3-dark .bp3-elevation-3{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
  
  .bp3-elevation-4{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2); }
    .bp3-elevation-4.bp3-dark,
    .bp3-dark .bp3-elevation-4{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }
  
  .bp3-card.bp3-interactive:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    cursor:pointer; }
    .bp3-card.bp3-interactive:hover.bp3-dark,
    .bp3-dark .bp3-card.bp3-interactive:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
  
  .bp3-card.bp3-interactive:active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
    opacity:0.9;
    -webkit-transition-duration:0;
            transition-duration:0; }
    .bp3-card.bp3-interactive:active.bp3-dark,
    .bp3-dark .bp3-card.bp3-interactive:active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  
  .bp3-collapse{
    height:0;
    overflow-y:hidden;
    -webkit-transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-collapse .bp3-collapse-body{
      -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-collapse .bp3-collapse-body[aria-hidden="true"]{
        display:none; }
  
  .bp3-context-menu .bp3-popover-target{
    display:block; }
  
  .bp3-context-menu-popover-target{
    position:fixed; }
  
  .bp3-divider{
    border-bottom:1px solid rgba(16, 22, 26, 0.15);
    border-right:1px solid rgba(16, 22, 26, 0.15);
    margin:5px; }
    .bp3-dark .bp3-divider{
      border-color:rgba(16, 22, 26, 0.4); }
  .bp3-dialog-container{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-pack:center;
        -ms-flex-pack:center;
            justify-content:center;
    min-height:100%;
    pointer-events:none;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none;
    width:100%; }
    .bp3-dialog-container.bp3-overlay-enter > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear > .bp3-dialog{
      opacity:0;
      -webkit-transform:scale(0.5);
              transform:scale(0.5); }
    .bp3-dialog-container.bp3-overlay-enter-active > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear-active > .bp3-dialog{
      opacity:1;
      -webkit-transform:scale(1);
              transform:scale(1);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:300ms;
              transition-duration:300ms;
      -webkit-transition-property:opacity, -webkit-transform;
      transition-property:opacity, -webkit-transform;
      transition-property:opacity, transform;
      transition-property:opacity, transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
              transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
    .bp3-dialog-container.bp3-overlay-exit > .bp3-dialog{
      opacity:1;
      -webkit-transform:scale(1);
              transform:scale(1); }
    .bp3-dialog-container.bp3-overlay-exit-active > .bp3-dialog{
      opacity:0;
      -webkit-transform:scale(0.5);
              transform:scale(0.5);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:300ms;
              transition-duration:300ms;
      -webkit-transition-property:opacity, -webkit-transform;
      transition-property:opacity, -webkit-transform;
      transition-property:opacity, transform;
      transition-property:opacity, transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
              transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  
  .bp3-dialog{
    background:#ebf1f5;
    border-radius:6px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    margin:30px 0;
    padding-bottom:20px;
    pointer-events:all;
    -webkit-user-select:text;
       -moz-user-select:text;
        -ms-user-select:text;
            user-select:text;
    width:500px; }
    .bp3-dialog:focus{
      outline:0; }
    .bp3-dialog.bp3-dark,
    .bp3-dark .bp3-dialog{
      background:#293742;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
  
  .bp3-dialog-header{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    background:#ffffff;
    border-radius:6px 6px 0 0;
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    min-height:40px;
    padding-left:20px;
    padding-right:5px; }
    .bp3-dialog-header .bp3-icon-large,
    .bp3-dialog-header .bp3-icon{
      color:#5c7080;
      -webkit-box-flex:0;
          -ms-flex:0 0 auto;
              flex:0 0 auto;
      margin-right:10px; }
    .bp3-dialog-header .bp3-heading{
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      word-wrap:normal;
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto;
      line-height:inherit;
      margin:0; }
      .bp3-dialog-header .bp3-heading:last-child{
        margin-right:20px; }
    .bp3-dark .bp3-dialog-header{
      background:#30404d;
      -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
              box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-dialog-header .bp3-icon-large,
      .bp3-dark .bp3-dialog-header .bp3-icon{
        color:#a7b6c2; }
  
  .bp3-dialog-body{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:18px;
    margin:20px; }
  
  .bp3-dialog-footer{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin:0 20px; }
  
  .bp3-dialog-footer-actions{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-pack:end;
        -ms-flex-pack:end;
            justify-content:flex-end; }
    .bp3-dialog-footer-actions .bp3-button{
      margin-left:10px; }
  .bp3-drawer{
    background:#ffffff;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    margin:0;
    padding:0; }
    .bp3-drawer:focus{
      outline:0; }
    .bp3-drawer.bp3-position-top{
      height:50%;
      left:0;
      right:0;
      top:0; }
      .bp3-drawer.bp3-position-top.bp3-overlay-enter, .bp3-drawer.bp3-position-top.bp3-overlay-appear{
        -webkit-transform:translateY(-100%);
                transform:translateY(-100%); }
      .bp3-drawer.bp3-position-top.bp3-overlay-enter-active, .bp3-drawer.bp3-position-top.bp3-overlay-appear-active{
        -webkit-transform:translateY(0);
                transform:translateY(0);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:200ms;
                transition-duration:200ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-drawer.bp3-position-top.bp3-overlay-exit{
        -webkit-transform:translateY(0);
                transform:translateY(0); }
      .bp3-drawer.bp3-position-top.bp3-overlay-exit-active{
        -webkit-transform:translateY(-100%);
                transform:translateY(-100%);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-bottom{
      bottom:0;
      height:50%;
      left:0;
      right:0; }
      .bp3-drawer.bp3-position-bottom.bp3-overlay-enter, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear{
        -webkit-transform:translateY(100%);
                transform:translateY(100%); }
      .bp3-drawer.bp3-position-bottom.bp3-overlay-enter-active, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear-active{
        -webkit-transform:translateY(0);
                transform:translateY(0);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:200ms;
                transition-duration:200ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-drawer.bp3-position-bottom.bp3-overlay-exit{
        -webkit-transform:translateY(0);
                transform:translateY(0); }
      .bp3-drawer.bp3-position-bottom.bp3-overlay-exit-active{
        -webkit-transform:translateY(100%);
                transform:translateY(100%);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-left{
      bottom:0;
      left:0;
      top:0;
      width:50%; }
      .bp3-drawer.bp3-position-left.bp3-overlay-enter, .bp3-drawer.bp3-position-left.bp3-overlay-appear{
        -webkit-transform:translateX(-100%);
                transform:translateX(-100%); }
      .bp3-drawer.bp3-position-left.bp3-overlay-enter-active, .bp3-drawer.bp3-position-left.bp3-overlay-appear-active{
        -webkit-transform:translateX(0);
                transform:translateX(0);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:200ms;
                transition-duration:200ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-drawer.bp3-position-left.bp3-overlay-exit{
        -webkit-transform:translateX(0);
                transform:translateX(0); }
      .bp3-drawer.bp3-position-left.bp3-overlay-exit-active{
        -webkit-transform:translateX(-100%);
                transform:translateX(-100%);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-right{
      bottom:0;
      right:0;
      top:0;
      width:50%; }
      .bp3-drawer.bp3-position-right.bp3-overlay-enter, .bp3-drawer.bp3-position-right.bp3-overlay-appear{
        -webkit-transform:translateX(100%);
                transform:translateX(100%); }
      .bp3-drawer.bp3-position-right.bp3-overlay-enter-active, .bp3-drawer.bp3-position-right.bp3-overlay-appear-active{
        -webkit-transform:translateX(0);
                transform:translateX(0);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:200ms;
                transition-duration:200ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-drawer.bp3-position-right.bp3-overlay-exit{
        -webkit-transform:translateX(0);
                transform:translateX(0); }
      .bp3-drawer.bp3-position-right.bp3-overlay-exit-active{
        -webkit-transform:translateX(100%);
                transform:translateX(100%);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical){
      bottom:0;
      right:0;
      top:0;
      width:50%; }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear{
        -webkit-transform:translateX(100%);
                transform:translateX(100%); }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear-active{
        -webkit-transform:translateX(0);
                transform:translateX(0);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:200ms;
                transition-duration:200ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit{
        -webkit-transform:translateX(0);
                transform:translateX(0); }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit-active{
        -webkit-transform:translateX(100%);
                transform:translateX(100%);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical{
      bottom:0;
      height:50%;
      left:0;
      right:0; }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right).bp3-vertical.bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right).bp3-vertical.bp3-overlay-appear{
        -webkit-transform:translateY(100%);
                transform:translateY(100%); }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right).bp3-vertical.bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right).bp3-vertical.bp3-overlay-appear-active{
        -webkit-transform:translateY(0);
                transform:translateY(0);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:200ms;
                transition-duration:200ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right).bp3-vertical.bp3-overlay-exit{
        -webkit-transform:translateY(0);
                transform:translateY(0); }
      .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
      .bp3-position-right).bp3-vertical.bp3-overlay-exit-active{
        -webkit-transform:translateY(100%);
                transform:translateY(100%);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-dark,
    .bp3-dark .bp3-drawer{
      background:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
  
  .bp3-drawer-header{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    border-radius:0;
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    min-height:40px;
    padding:5px;
    padding-left:20px;
    position:relative; }
    .bp3-drawer-header .bp3-icon-large,
    .bp3-drawer-header .bp3-icon{
      color:#5c7080;
      -webkit-box-flex:0;
          -ms-flex:0 0 auto;
              flex:0 0 auto;
      margin-right:10px; }
    .bp3-drawer-header .bp3-heading{
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      word-wrap:normal;
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto;
      line-height:inherit;
      margin:0; }
      .bp3-drawer-header .bp3-heading:last-child{
        margin-right:20px; }
    .bp3-dark .bp3-drawer-header{
      -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
              box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-drawer-header .bp3-icon-large,
      .bp3-dark .bp3-drawer-header .bp3-icon{
        color:#a7b6c2; }
  
  .bp3-drawer-body{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:18px;
    overflow:auto; }
  
  .bp3-drawer-footer{
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    padding:10px 20px;
    position:relative; }
    .bp3-dark .bp3-drawer-footer{
      -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4); }
  .bp3-editable-text{
    cursor:text;
    display:inline-block;
    max-width:100%;
    position:relative;
    vertical-align:top;
    white-space:nowrap; }
    .bp3-editable-text::before{
      bottom:-3px;
      left:-3px;
      position:absolute;
      right:-3px;
      top:-3px;
      border-radius:3px;
      content:"";
      -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-editable-text:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
    .bp3-editable-text.bp3-editable-text-editing::before{
      background-color:#ffffff;
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-editable-text.bp3-disabled::before{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-editable-text.bp3-intent-primary .bp3-editable-text-input,
    .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
      color:#137cbd; }
    .bp3-editable-text.bp3-intent-primary:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4); }
    .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-editable-text.bp3-intent-success .bp3-editable-text-input,
    .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
      color:#0f9960; }
    .bp3-editable-text.bp3-intent-success:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4); }
    .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-editable-text.bp3-intent-warning .bp3-editable-text-input,
    .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
      color:#d9822b; }
    .bp3-editable-text.bp3-intent-warning:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4); }
    .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-editable-text.bp3-intent-danger .bp3-editable-text-input,
    .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
      color:#db3737; }
    .bp3-editable-text.bp3-intent-danger:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4); }
    .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-editable-text:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15); }
    .bp3-dark .bp3-editable-text.bp3-editable-text-editing::before{
      background-color:rgba(16, 22, 26, 0.3);
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-disabled::before{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
      color:#48aff0; }
    .bp3-dark .bp3-editable-text.bp3-intent-primary:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4);
              box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
      color:#3dcc91; }
    .bp3-dark .bp3-editable-text.bp3-intent-success:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4);
              box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
      color:#ffb366; }
    .bp3-dark .bp3-editable-text.bp3-intent-warning:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4);
              box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
      color:#ff7373; }
    .bp3-dark .bp3-editable-text.bp3-intent-danger:hover::before{
      -webkit-box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4);
              box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4); }
    .bp3-dark .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
      -webkit-box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  
  .bp3-editable-text-input,
  .bp3-editable-text-content{
    color:inherit;
    display:inherit;
    font:inherit;
    letter-spacing:inherit;
    max-width:inherit;
    min-width:inherit;
    position:relative;
    resize:none;
    text-transform:inherit;
    vertical-align:top; }
  
  .bp3-editable-text-input{
    background:none;
    border:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    padding:0;
    white-space:pre-wrap;
    width:100%; }
    .bp3-editable-text-input::-webkit-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-editable-text-input::-moz-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-editable-text-input:-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-editable-text-input::-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-editable-text-input::placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-editable-text-input:focus{
      outline:none; }
    .bp3-editable-text-input::-ms-clear{
      display:none; }
  
  .bp3-editable-text-content{
    overflow:hidden;
    padding-right:2px;
    text-overflow:ellipsis;
    white-space:pre; }
    .bp3-editable-text-editing > .bp3-editable-text-content{
      left:0;
      position:absolute;
      visibility:hidden; }
    .bp3-editable-text-placeholder > .bp3-editable-text-content{
      color:rgba(92, 112, 128, 0.6); }
      .bp3-dark .bp3-editable-text-placeholder > .bp3-editable-text-content{
        color:rgba(167, 182, 194, 0.6); }
  
  .bp3-editable-text.bp3-multiline{
    display:block; }
    .bp3-editable-text.bp3-multiline .bp3-editable-text-content{
      overflow:auto;
      white-space:pre-wrap;
      word-wrap:break-word; }
  .bp3-divider{
    border-bottom:1px solid rgba(16, 22, 26, 0.15);
    border-right:1px solid rgba(16, 22, 26, 0.15);
    margin:5px; }
    .bp3-dark .bp3-divider{
      border-color:rgba(16, 22, 26, 0.4); }
  .bp3-control-group{
    -webkit-transform:translateZ(0);
            transform:translateZ(0);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch; }
    .bp3-control-group > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-control-group > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-control-group .bp3-button,
    .bp3-control-group .bp3-html-select,
    .bp3-control-group .bp3-input,
    .bp3-control-group .bp3-select{
      position:relative; }
    .bp3-control-group .bp3-input{
      border-radius:inherit;
      z-index:2; }
      .bp3-control-group .bp3-input:focus{
        border-radius:3px;
        z-index:14; }
      .bp3-control-group .bp3-input[class*="bp3-intent"]{
        z-index:13; }
        .bp3-control-group .bp3-input[class*="bp3-intent"]:focus{
          z-index:15; }
      .bp3-control-group .bp3-input[readonly], .bp3-control-group .bp3-input:disabled, .bp3-control-group .bp3-input.bp3-disabled{
        z-index:1; }
    .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input{
      z-index:13; }
      .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input:focus{
        z-index:15; }
    .bp3-control-group .bp3-button,
    .bp3-control-group .bp3-html-select select,
    .bp3-control-group .bp3-select select{
      -webkit-transform:translateZ(0);
              transform:translateZ(0);
      border-radius:inherit;
      z-index:4; }
      .bp3-control-group .bp3-button:focus,
      .bp3-control-group .bp3-html-select select:focus,
      .bp3-control-group .bp3-select select:focus{
        z-index:5; }
      .bp3-control-group .bp3-button:hover,
      .bp3-control-group .bp3-html-select select:hover,
      .bp3-control-group .bp3-select select:hover{
        z-index:6; }
      .bp3-control-group .bp3-button:active,
      .bp3-control-group .bp3-html-select select:active,
      .bp3-control-group .bp3-select select:active{
        z-index:7; }
      .bp3-control-group .bp3-button[readonly], .bp3-control-group .bp3-button:disabled, .bp3-control-group .bp3-button.bp3-disabled,
      .bp3-control-group .bp3-html-select select[readonly],
      .bp3-control-group .bp3-html-select select:disabled,
      .bp3-control-group .bp3-html-select select.bp3-disabled,
      .bp3-control-group .bp3-select select[readonly],
      .bp3-control-group .bp3-select select:disabled,
      .bp3-control-group .bp3-select select.bp3-disabled{
        z-index:3; }
      .bp3-control-group .bp3-button[class*="bp3-intent"],
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"],
      .bp3-control-group .bp3-select select[class*="bp3-intent"]{
        z-index:9; }
        .bp3-control-group .bp3-button[class*="bp3-intent"]:focus,
        .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:focus,
        .bp3-control-group .bp3-select select[class*="bp3-intent"]:focus{
          z-index:10; }
        .bp3-control-group .bp3-button[class*="bp3-intent"]:hover,
        .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:hover,
        .bp3-control-group .bp3-select select[class*="bp3-intent"]:hover{
          z-index:11; }
        .bp3-control-group .bp3-button[class*="bp3-intent"]:active,
        .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:active,
        .bp3-control-group .bp3-select select[class*="bp3-intent"]:active{
          z-index:12; }
        .bp3-control-group .bp3-button[class*="bp3-intent"][readonly], .bp3-control-group .bp3-button[class*="bp3-intent"]:disabled, .bp3-control-group .bp3-button[class*="bp3-intent"].bp3-disabled,
        .bp3-control-group .bp3-html-select select[class*="bp3-intent"][readonly],
        .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:disabled,
        .bp3-control-group .bp3-html-select select[class*="bp3-intent"].bp3-disabled,
        .bp3-control-group .bp3-select select[class*="bp3-intent"][readonly],
        .bp3-control-group .bp3-select select[class*="bp3-intent"]:disabled,
        .bp3-control-group .bp3-select select[class*="bp3-intent"].bp3-disabled{
          z-index:8; }
    .bp3-control-group .bp3-input-group > .bp3-icon,
    .bp3-control-group .bp3-input-group > .bp3-button,
    .bp3-control-group .bp3-input-group > .bp3-input-action{
      z-index:16; }
    .bp3-control-group .bp3-select::after,
    .bp3-control-group .bp3-html-select::after,
    .bp3-control-group .bp3-select > .bp3-icon,
    .bp3-control-group .bp3-html-select > .bp3-icon{
      z-index:17; }
    .bp3-control-group .bp3-select:focus-within{
      z-index:5; }
    .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
      margin-right:-1px; }
    .bp3-control-group:not(.bp3-vertical) > .bp3-divider:not(:first-child){
      margin-left:6px; }
    .bp3-dark .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
      margin-right:0; }
    .bp3-dark .bp3-control-group:not(.bp3-vertical) > .bp3-button + .bp3-button{
      margin-left:1px; }
    .bp3-control-group .bp3-popover-wrapper,
    .bp3-control-group .bp3-popover-target{
      border-radius:inherit; }
    .bp3-control-group > :first-child{
      border-radius:3px 0 0 3px; }
    .bp3-control-group > :last-child{
      border-radius:0 3px 3px 0;
      margin-right:0; }
    .bp3-control-group > :only-child{
      border-radius:3px;
      margin-right:0; }
    .bp3-control-group .bp3-input-group .bp3-button{
      border-radius:3px; }
    .bp3-control-group .bp3-numeric-input:not(:first-child) .bp3-input-group{
      border-bottom-left-radius:0;
      border-top-left-radius:0; }
    .bp3-control-group.bp3-fill{
      width:100%; }
    .bp3-control-group > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto; }
    .bp3-control-group.bp3-fill > *:not(.bp3-fixed){
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto; }
    .bp3-control-group.bp3-vertical{
      -webkit-box-orient:vertical;
      -webkit-box-direction:normal;
          -ms-flex-direction:column;
              flex-direction:column; }
      .bp3-control-group.bp3-vertical > *{
        margin-top:-1px; }
      .bp3-control-group.bp3-vertical > :first-child{
        border-radius:3px 3px 0 0;
        margin-top:0; }
      .bp3-control-group.bp3-vertical > :last-child{
        border-radius:0 0 3px 3px; }
  .bp3-control{
    cursor:pointer;
    display:block;
    margin-bottom:10px;
    position:relative;
    text-transform:none; }
    .bp3-control input:checked ~ .bp3-control-indicator{
      background-color:#137cbd;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      color:#ffffff; }
    .bp3-control:hover input:checked ~ .bp3-control-indicator{
      background-color:#106ba3;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
      background:#0e5a8a;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-control input:disabled:checked ~ .bp3-control-indicator{
      background:rgba(19, 124, 189, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-control input:checked ~ .bp3-control-indicator{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control:hover input:checked ~ .bp3-control-indicator{
      background-color:#106ba3;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
      background-color:#0e5a8a;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-control input:disabled:checked ~ .bp3-control-indicator{
      background:rgba(14, 90, 138, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-control:not(.bp3-align-right){
      padding-left:26px; }
      .bp3-control:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-26px; }
    .bp3-control.bp3-align-right{
      padding-right:26px; }
      .bp3-control.bp3-align-right .bp3-control-indicator{
        margin-right:-26px; }
    .bp3-control.bp3-disabled{
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
    .bp3-control.bp3-inline{
      display:inline-block;
      margin-right:20px; }
    .bp3-control input{
      left:0;
      opacity:0;
      position:absolute;
      top:0;
      z-index:-1; }
    .bp3-control .bp3-control-indicator{
      background-clip:padding-box;
      background-color:#f5f8fa;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
      border:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      cursor:pointer;
      display:inline-block;
      font-size:16px;
      height:1em;
      margin-right:10px;
      margin-top:-3px;
      position:relative;
      -webkit-user-select:none;
         -moz-user-select:none;
          -ms-user-select:none;
              user-select:none;
      vertical-align:middle;
      width:1em; }
      .bp3-control .bp3-control-indicator::before{
        content:"";
        display:block;
        height:1em;
        width:1em; }
    .bp3-control:hover .bp3-control-indicator{
      background-color:#ebf1f5; }
    .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
      background:#d8e1e8;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-control input:disabled ~ .bp3-control-indicator{
      background:rgba(206, 217, 224, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      cursor:not-allowed; }
    .bp3-control input:focus ~ .bp3-control-indicator{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:2px;
      -moz-outline-radius:6px; }
    .bp3-control.bp3-align-right .bp3-control-indicator{
      float:right;
      margin-left:10px;
      margin-top:1px; }
    .bp3-control.bp3-large{
      font-size:16px; }
      .bp3-control.bp3-large:not(.bp3-align-right){
        padding-left:30px; }
        .bp3-control.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
          margin-left:-30px; }
      .bp3-control.bp3-large.bp3-align-right{
        padding-right:30px; }
        .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
          margin-right:-30px; }
      .bp3-control.bp3-large .bp3-control-indicator{
        font-size:20px; }
      .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-top:0; }
    .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
      background-color:#137cbd;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
      color:#ffffff; }
    .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
      background-color:#106ba3;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
      background:#0e5a8a;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      background:rgba(19, 124, 189, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
      background-color:#106ba3;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
      background-color:#0e5a8a;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      background:rgba(14, 90, 138, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-control.bp3-checkbox .bp3-control-indicator{
      border-radius:3px; }
    .bp3-control.bp3-checkbox input:checked ~ .bp3-control-indicator::before{
      background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 00-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0012 5z' fill='white'/%3e%3c/svg%3e"); }
    .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator::before{
      background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z' fill='white'/%3e%3c/svg%3e"); }
    .bp3-control.bp3-radio .bp3-control-indicator{
      border-radius:50%; }
    .bp3-control.bp3-radio input:checked ~ .bp3-control-indicator::before{
      background-image:radial-gradient(#ffffff, #ffffff 28%, transparent 32%); }
    .bp3-control.bp3-radio input:checked:disabled ~ .bp3-control-indicator::before{
      opacity:0.5; }
    .bp3-control.bp3-radio input:focus ~ .bp3-control-indicator{
      -moz-outline-radius:16px; }
    .bp3-control.bp3-switch input ~ .bp3-control-indicator{
      background:rgba(167, 182, 194, 0.5); }
    .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
      background:rgba(115, 134, 148, 0.5); }
    .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
      background:rgba(92, 112, 128, 0.5); }
    .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
      background:rgba(206, 217, 224, 0.5); }
      .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
        background:rgba(255, 255, 255, 0.8); }
    .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
      background:#137cbd; }
    .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
      background:#106ba3; }
    .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
      background:#0e5a8a; }
    .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
      background:rgba(19, 124, 189, 0.5); }
      .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
        background:rgba(255, 255, 255, 0.8); }
    .bp3-control.bp3-switch:not(.bp3-align-right){
      padding-left:38px; }
      .bp3-control.bp3-switch:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-38px; }
    .bp3-control.bp3-switch.bp3-align-right{
      padding-right:38px; }
      .bp3-control.bp3-switch.bp3-align-right .bp3-control-indicator{
        margin-right:-38px; }
    .bp3-control.bp3-switch .bp3-control-indicator{
      border:none;
      border-radius:1.75em;
      -webkit-box-shadow:none !important;
              box-shadow:none !important;
      min-width:1.75em;
      -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      width:auto; }
      .bp3-control.bp3-switch .bp3-control-indicator::before{
        background:#ffffff;
        border-radius:50%;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
        height:calc(1em - 4px);
        left:0;
        margin:2px;
        position:absolute;
        -webkit-transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
        transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
        width:calc(1em - 4px); }
    .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
      left:calc(100% - 1em); }
    .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right){
      padding-left:45px; }
      .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-45px; }
    .bp3-control.bp3-switch.bp3-large.bp3-align-right{
      padding-right:45px; }
      .bp3-control.bp3-switch.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-right:-45px; }
    .bp3-dark .bp3-control.bp3-switch input ~ .bp3-control-indicator{
      background:rgba(16, 22, 26, 0.5); }
    .bp3-dark .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
      background:rgba(16, 22, 26, 0.7); }
    .bp3-dark .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
      background:rgba(16, 22, 26, 0.9); }
    .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
      background:rgba(57, 75, 89, 0.5); }
      .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
        background:rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
      background:#137cbd; }
    .bp3-dark .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
      background:#106ba3; }
    .bp3-dark .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
      background:#0e5a8a; }
    .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
      background:rgba(14, 90, 138, 0.5); }
      .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
        background:rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control.bp3-switch .bp3-control-indicator::before{
      background:#394b59;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-control.bp3-switch .bp3-switch-inner-text{
      font-size:0.7em;
      text-align:center; }
    .bp3-control.bp3-switch .bp3-control-indicator-child:first-child{
      line-height:0;
      margin-left:0.5em;
      margin-right:1.2em;
      visibility:hidden; }
    .bp3-control.bp3-switch .bp3-control-indicator-child:last-child{
      line-height:1em;
      margin-left:1.2em;
      margin-right:0.5em;
      visibility:visible; }
    .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:first-child{
      line-height:1em;
      visibility:visible; }
    .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:last-child{
      line-height:0;
      visibility:hidden; }
    .bp3-dark .bp3-control{
      color:#f5f8fa; }
      .bp3-dark .bp3-control.bp3-disabled{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-control .bp3-control-indicator{
        background-color:#394b59;
        background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
        background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-control:hover .bp3-control-indicator{
        background-color:#30404d; }
      .bp3-dark .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
        background:#202b33;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-control input:disabled ~ .bp3-control-indicator{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        cursor:not-allowed; }
      .bp3-dark .bp3-control.bp3-checkbox input:disabled:checked ~ .bp3-control-indicator, .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
        color:rgba(167, 182, 194, 0.6); }
  .bp3-file-input{
    cursor:pointer;
    display:inline-block;
    height:30px;
    position:relative; }
    .bp3-file-input input{
      margin:0;
      min-width:200px;
      opacity:0; }
      .bp3-file-input input:disabled + .bp3-file-upload-input,
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
        background:rgba(206, 217, 224, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed;
        resize:none; }
        .bp3-file-input input:disabled + .bp3-file-upload-input::after,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
          background-color:rgba(206, 217, 224, 0.5);
          background-image:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:rgba(92, 112, 128, 0.6);
          cursor:not-allowed;
          outline:none; }
          .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active:hover,
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active,
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active:hover{
            background:rgba(206, 217, 224, 0.7); }
        .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input, .bp3-dark
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
          background:rgba(57, 75, 89, 0.5);
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after, .bp3-dark
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
            background-color:rgba(57, 75, 89, 0.5);
            background-image:none;
            -webkit-box-shadow:none;
                    box-shadow:none;
            color:rgba(167, 182, 194, 0.6); }
            .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-dark
            .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active{
              background:rgba(57, 75, 89, 0.7); }
    .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
      color:#182026; }
    .bp3-dark .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
      color:#f5f8fa; }
    .bp3-file-input.bp3-fill{
      width:100%; }
    .bp3-file-input.bp3-large,
    .bp3-large .bp3-file-input{
      height:40px; }
    .bp3-file-input .bp3-file-upload-input-custom-text::after{
      content:attr(bp3-button-text); }
  
  .bp3-file-upload-input{
    -webkit-appearance:none;
       -moz-appearance:none;
            appearance:none;
    background:#ffffff;
    border:none;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
    color:#182026;
    font-size:14px;
    font-weight:400;
    height:30px;
    line-height:30px;
    outline:none;
    padding:0 10px;
    -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    vertical-align:middle;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    color:rgba(92, 112, 128, 0.6);
    left:0;
    padding-right:80px;
    position:absolute;
    right:0;
    top:0;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-file-upload-input::-webkit-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-file-upload-input::-moz-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-file-upload-input:-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-file-upload-input::-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-file-upload-input::placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-file-upload-input:focus, .bp3-file-upload-input.bp3-active{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-file-upload-input[type="search"], .bp3-file-upload-input.bp3-round{
      border-radius:30px;
      -webkit-box-sizing:border-box;
              box-sizing:border-box;
      padding-left:10px; }
    .bp3-file-upload-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
    .bp3-file-upload-input:disabled, .bp3-file-upload-input.bp3-disabled{
      background:rgba(206, 217, 224, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      resize:none; }
    .bp3-file-upload-input::after{
      background-color:#f5f8fa;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      color:#182026;
      min-height:24px;
      min-width:24px;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      word-wrap:normal;
      border-radius:3px;
      content:"Browse";
      line-height:24px;
      margin:3px;
      position:absolute;
      right:0;
      text-align:center;
      top:0;
      width:70px; }
      .bp3-file-upload-input::after:hover{
        background-clip:padding-box;
        background-color:#ebf1f5;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
      .bp3-file-upload-input::after:active, .bp3-file-upload-input::after.bp3-active{
        background-color:#d8e1e8;
        background-image:none;
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-file-upload-input::after:disabled, .bp3-file-upload-input::after.bp3-disabled{
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed;
        outline:none; }
        .bp3-file-upload-input::after:disabled.bp3-active, .bp3-file-upload-input::after:disabled.bp3-active:hover, .bp3-file-upload-input::after.bp3-disabled.bp3-active, .bp3-file-upload-input::after.bp3-disabled.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
    .bp3-file-upload-input:hover::after{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-file-upload-input:active::after{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-large .bp3-file-upload-input{
      font-size:16px;
      height:40px;
      line-height:40px;
      padding-right:95px; }
      .bp3-large .bp3-file-upload-input[type="search"], .bp3-large .bp3-file-upload-input.bp3-round{
        padding:0 15px; }
      .bp3-large .bp3-file-upload-input::after{
        min-height:30px;
        min-width:30px;
        line-height:30px;
        margin:5px;
        width:85px; }
    .bp3-dark .bp3-file-upload-input{
      background:rgba(16, 22, 26, 0.3);
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-file-upload-input::-webkit-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-file-upload-input::-moz-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-file-upload-input:-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-file-upload-input::-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-file-upload-input::placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-file-upload-input:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-file-upload-input[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-file-upload-input:disabled, .bp3-dark .bp3-file-upload-input.bp3-disabled{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-file-upload-input::after{
        background-color:#394b59;
        background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
        background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
        color:#f5f8fa; }
        .bp3-dark .bp3-file-upload-input::after:hover, .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
          color:#f5f8fa; }
        .bp3-dark .bp3-file-upload-input::after:hover{
          background-color:#30404d;
          -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                  box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
          background-color:#202b33;
          background-image:none;
          -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                  box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
        .bp3-dark .bp3-file-upload-input::after:disabled, .bp3-dark .bp3-file-upload-input::after.bp3-disabled{
          background-color:rgba(57, 75, 89, 0.5);
          background-image:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-upload-input::after:disabled.bp3-active, .bp3-dark .bp3-file-upload-input::after.bp3-disabled.bp3-active{
            background:rgba(57, 75, 89, 0.7); }
        .bp3-dark .bp3-file-upload-input::after .bp3-button-spinner .bp3-spinner-head{
          background:rgba(16, 22, 26, 0.5);
          stroke:#8a9ba8; }
      .bp3-dark .bp3-file-upload-input:hover::after{
        background-color:#30404d;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-file-upload-input:active::after{
        background-color:#202b33;
        background-image:none;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-file-upload-input::after{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-form-group{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    margin:0 0 15px; }
    .bp3-form-group label.bp3-label{
      margin-bottom:5px; }
    .bp3-form-group .bp3-control{
      margin-top:7px; }
    .bp3-form-group .bp3-form-helper-text{
      color:#5c7080;
      font-size:12px;
      margin-top:5px; }
    .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
      color:#106ba3; }
    .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
      color:#0d8050; }
    .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
      color:#bf7326; }
    .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
      color:#c23030; }
    .bp3-form-group.bp3-inline{
      -webkit-box-align:start;
          -ms-flex-align:start;
              align-items:flex-start;
      -webkit-box-orient:horizontal;
      -webkit-box-direction:normal;
          -ms-flex-direction:row;
              flex-direction:row; }
      .bp3-form-group.bp3-inline.bp3-large label.bp3-label{
        line-height:40px;
        margin:0 10px 0 0; }
      .bp3-form-group.bp3-inline label.bp3-label{
        line-height:30px;
        margin:0 10px 0 0; }
    .bp3-form-group.bp3-disabled .bp3-label,
    .bp3-form-group.bp3-disabled .bp3-text-muted,
    .bp3-form-group.bp3-disabled .bp3-form-helper-text{
      color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-dark .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
      color:#48aff0; }
    .bp3-dark .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
      color:#3dcc91; }
    .bp3-dark .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
      color:#ffb366; }
    .bp3-dark .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
      color:#ff7373; }
    .bp3-dark .bp3-form-group .bp3-form-helper-text{
      color:#a7b6c2; }
    .bp3-dark .bp3-form-group.bp3-disabled .bp3-label,
    .bp3-dark .bp3-form-group.bp3-disabled .bp3-text-muted,
    .bp3-dark .bp3-form-group.bp3-disabled .bp3-form-helper-text{
      color:rgba(167, 182, 194, 0.6) !important; }
  .bp3-input-group{
    display:block;
    position:relative; }
    .bp3-input-group .bp3-input{
      position:relative;
      width:100%; }
      .bp3-input-group .bp3-input:not(:first-child){
        padding-left:30px; }
      .bp3-input-group .bp3-input:not(:last-child){
        padding-right:30px; }
    .bp3-input-group .bp3-input-action,
    .bp3-input-group > .bp3-input-left-container,
    .bp3-input-group > .bp3-button,
    .bp3-input-group > .bp3-icon{
      position:absolute;
      top:0; }
      .bp3-input-group .bp3-input-action:first-child,
      .bp3-input-group > .bp3-input-left-container:first-child,
      .bp3-input-group > .bp3-button:first-child,
      .bp3-input-group > .bp3-icon:first-child{
        left:0; }
      .bp3-input-group .bp3-input-action:last-child,
      .bp3-input-group > .bp3-input-left-container:last-child,
      .bp3-input-group > .bp3-button:last-child,
      .bp3-input-group > .bp3-icon:last-child{
        right:0; }
    .bp3-input-group .bp3-button{
      min-height:24px;
      min-width:24px;
      margin:3px;
      padding:0 7px; }
      .bp3-input-group .bp3-button:empty{
        padding:0; }
    .bp3-input-group > .bp3-input-left-container,
    .bp3-input-group > .bp3-icon{
      z-index:1; }
    .bp3-input-group > .bp3-input-left-container > .bp3-icon,
    .bp3-input-group > .bp3-icon{
      color:#5c7080; }
      .bp3-input-group > .bp3-input-left-container > .bp3-icon:empty,
      .bp3-input-group > .bp3-icon:empty{
        font-family:"Icons16", sans-serif;
        font-size:16px;
        font-style:normal;
        font-weight:400;
        line-height:1;
        -moz-osx-font-smoothing:grayscale;
        -webkit-font-smoothing:antialiased; }
    .bp3-input-group > .bp3-input-left-container > .bp3-icon,
    .bp3-input-group > .bp3-icon,
    .bp3-input-group .bp3-input-action > .bp3-spinner{
      margin:7px; }
    .bp3-input-group .bp3-tag{
      margin:5px; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus),
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
      color:#5c7080; }
      .bp3-dark .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus), .bp3-dark
      .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
        color:#a7b6c2; }
      .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large,
      .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon,
      .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard,
      .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large{
        color:#5c7080; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled{
      color:rgba(92, 112, 128, 0.6) !important; }
      .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-large,
      .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon,
      .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-standard,
      .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-large{
        color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-input-group.bp3-disabled{
      cursor:not-allowed; }
      .bp3-input-group.bp3-disabled .bp3-icon{
        color:rgba(92, 112, 128, 0.6); }
    .bp3-input-group.bp3-large .bp3-button{
      min-height:30px;
      min-width:30px;
      margin:5px; }
    .bp3-input-group.bp3-large > .bp3-input-left-container > .bp3-icon,
    .bp3-input-group.bp3-large > .bp3-icon,
    .bp3-input-group.bp3-large .bp3-input-action > .bp3-spinner{
      margin:12px; }
    .bp3-input-group.bp3-large .bp3-input{
      font-size:16px;
      height:40px;
      line-height:40px; }
      .bp3-input-group.bp3-large .bp3-input[type="search"], .bp3-input-group.bp3-large .bp3-input.bp3-round{
        padding:0 15px; }
      .bp3-input-group.bp3-large .bp3-input:not(:first-child){
        padding-left:40px; }
      .bp3-input-group.bp3-large .bp3-input:not(:last-child){
        padding-right:40px; }
    .bp3-input-group.bp3-small .bp3-button{
      min-height:20px;
      min-width:20px;
      margin:2px; }
    .bp3-input-group.bp3-small .bp3-tag{
      min-height:20px;
      min-width:20px;
      margin:2px; }
    .bp3-input-group.bp3-small > .bp3-input-left-container > .bp3-icon,
    .bp3-input-group.bp3-small > .bp3-icon,
    .bp3-input-group.bp3-small .bp3-input-action > .bp3-spinner{
      margin:4px; }
    .bp3-input-group.bp3-small .bp3-input{
      font-size:12px;
      height:24px;
      line-height:24px;
      padding-left:8px;
      padding-right:8px; }
      .bp3-input-group.bp3-small .bp3-input[type="search"], .bp3-input-group.bp3-small .bp3-input.bp3-round{
        padding:0 12px; }
      .bp3-input-group.bp3-small .bp3-input:not(:first-child){
        padding-left:24px; }
      .bp3-input-group.bp3-small .bp3-input:not(:last-child){
        padding-right:24px; }
    .bp3-input-group.bp3-fill{
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto;
      width:100%; }
    .bp3-input-group.bp3-round .bp3-button,
    .bp3-input-group.bp3-round .bp3-input,
    .bp3-input-group.bp3-round .bp3-tag{
      border-radius:30px; }
    .bp3-dark .bp3-input-group .bp3-icon{
      color:#a7b6c2; }
    .bp3-dark .bp3-input-group.bp3-disabled .bp3-icon{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-input-group.bp3-intent-primary .bp3-input{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-primary .bp3-input:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-primary .bp3-input[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-input-group.bp3-intent-primary .bp3-input:disabled, .bp3-input-group.bp3-intent-primary .bp3-input.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-input-group.bp3-intent-primary > .bp3-icon{
      color:#106ba3; }
      .bp3-dark .bp3-input-group.bp3-intent-primary > .bp3-icon{
        color:#48aff0; }
    .bp3-input-group.bp3-intent-success .bp3-input{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-success .bp3-input:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-success .bp3-input[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-input-group.bp3-intent-success .bp3-input:disabled, .bp3-input-group.bp3-intent-success .bp3-input.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-input-group.bp3-intent-success > .bp3-icon{
      color:#0d8050; }
      .bp3-dark .bp3-input-group.bp3-intent-success > .bp3-icon{
        color:#3dcc91; }
    .bp3-input-group.bp3-intent-warning .bp3-input{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-warning .bp3-input:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-warning .bp3-input[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-input-group.bp3-intent-warning .bp3-input:disabled, .bp3-input-group.bp3-intent-warning .bp3-input.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-input-group.bp3-intent-warning > .bp3-icon{
      color:#bf7326; }
      .bp3-dark .bp3-input-group.bp3-intent-warning > .bp3-icon{
        color:#ffb366; }
    .bp3-input-group.bp3-intent-danger .bp3-input{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-danger .bp3-input:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input-group.bp3-intent-danger .bp3-input[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-input-group.bp3-intent-danger .bp3-input:disabled, .bp3-input-group.bp3-intent-danger .bp3-input.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-input-group.bp3-intent-danger > .bp3-icon{
      color:#c23030; }
      .bp3-dark .bp3-input-group.bp3-intent-danger > .bp3-icon{
        color:#ff7373; }
  .bp3-input{
    -webkit-appearance:none;
       -moz-appearance:none;
            appearance:none;
    background:#ffffff;
    border:none;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
    color:#182026;
    font-size:14px;
    font-weight:400;
    height:30px;
    line-height:30px;
    outline:none;
    padding:0 10px;
    -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    vertical-align:middle; }
    .bp3-input::-webkit-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input::-moz-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input:-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input::-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input::placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input:focus, .bp3-input.bp3-active{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input[type="search"], .bp3-input.bp3-round{
      border-radius:30px;
      -webkit-box-sizing:border-box;
              box-sizing:border-box;
      padding-left:10px; }
    .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
    .bp3-input:disabled, .bp3-input.bp3-disabled{
      background:rgba(206, 217, 224, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      resize:none; }
    .bp3-input.bp3-large{
      font-size:16px;
      height:40px;
      line-height:40px; }
      .bp3-input.bp3-large[type="search"], .bp3-input.bp3-large.bp3-round{
        padding:0 15px; }
    .bp3-input.bp3-small{
      font-size:12px;
      height:24px;
      line-height:24px;
      padding-left:8px;
      padding-right:8px; }
      .bp3-input.bp3-small[type="search"], .bp3-input.bp3-small.bp3-round{
        padding:0 12px; }
    .bp3-input.bp3-fill{
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto;
      width:100%; }
    .bp3-dark .bp3-input{
      background:rgba(16, 22, 26, 0.3);
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark .bp3-input::-webkit-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-input::-moz-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-input:-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-input::-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-input::placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-input:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input:disabled, .bp3-dark .bp3-input.bp3-disabled{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
    .bp3-input.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-primary:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-primary[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-input.bp3-intent-primary:disabled, .bp3-input.bp3-intent-primary.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-input.bp3-intent-primary{
        -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-primary:focus{
          -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                  box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-primary[readonly]{
          -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                  box-shadow:inset 0 0 0 1px #137cbd; }
        .bp3-dark .bp3-input.bp3-intent-primary:disabled, .bp3-dark .bp3-input.bp3-intent-primary.bp3-disabled{
          -webkit-box-shadow:none;
                  box-shadow:none; }
    .bp3-input.bp3-intent-success{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-success:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-success[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-input.bp3-intent-success:disabled, .bp3-input.bp3-intent-success.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-input.bp3-intent-success{
        -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-success:focus{
          -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                  box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-success[readonly]{
          -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                  box-shadow:inset 0 0 0 1px #0f9960; }
        .bp3-dark .bp3-input.bp3-intent-success:disabled, .bp3-dark .bp3-input.bp3-intent-success.bp3-disabled{
          -webkit-box-shadow:none;
                  box-shadow:none; }
    .bp3-input.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-warning:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-warning[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-input.bp3-intent-warning:disabled, .bp3-input.bp3-intent-warning.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-input.bp3-intent-warning{
        -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-warning:focus{
          -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                  box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-warning[readonly]{
          -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                  box-shadow:inset 0 0 0 1px #d9822b; }
        .bp3-dark .bp3-input.bp3-intent-warning:disabled, .bp3-dark .bp3-input.bp3-intent-warning.bp3-disabled{
          -webkit-box-shadow:none;
                  box-shadow:none; }
    .bp3-input.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-danger:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-input.bp3-intent-danger[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-input.bp3-intent-danger:disabled, .bp3-input.bp3-intent-danger.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-input.bp3-intent-danger{
        -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-danger:focus{
          -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                  box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
        .bp3-dark .bp3-input.bp3-intent-danger[readonly]{
          -webkit-box-shadow:inset 0 0 0 1px #db3737;
                  box-shadow:inset 0 0 0 1px #db3737; }
        .bp3-dark .bp3-input.bp3-intent-danger:disabled, .bp3-dark .bp3-input.bp3-intent-danger.bp3-disabled{
          -webkit-box-shadow:none;
                  box-shadow:none; }
    .bp3-input::-ms-clear{
      display:none; }
  textarea.bp3-input{
    max-width:100%;
    padding:10px; }
    textarea.bp3-input, textarea.bp3-input.bp3-large, textarea.bp3-input.bp3-small{
      height:auto;
      line-height:inherit; }
    textarea.bp3-input.bp3-small{
      padding:8px; }
    .bp3-dark textarea.bp3-input{
      background:rgba(16, 22, 26, 0.3);
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark textarea.bp3-input::-webkit-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark textarea.bp3-input::-moz-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark textarea.bp3-input:-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark textarea.bp3-input::-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark textarea.bp3-input::placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark textarea.bp3-input:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark textarea.bp3-input[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark textarea.bp3-input:disabled, .bp3-dark textarea.bp3-input.bp3-disabled{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
  label.bp3-label{
    display:block;
    margin-bottom:15px;
    margin-top:0; }
    label.bp3-label .bp3-html-select,
    label.bp3-label .bp3-input,
    label.bp3-label .bp3-select,
    label.bp3-label .bp3-slider,
    label.bp3-label .bp3-popover-wrapper{
      display:block;
      margin-top:5px;
      text-transform:none; }
    label.bp3-label .bp3-button-group{
      margin-top:5px; }
    label.bp3-label .bp3-select select,
    label.bp3-label .bp3-html-select select{
      font-weight:400;
      vertical-align:top;
      width:100%; }
    label.bp3-label.bp3-disabled,
    label.bp3-label.bp3-disabled .bp3-text-muted{
      color:rgba(92, 112, 128, 0.6); }
    label.bp3-label.bp3-inline{
      line-height:30px; }
      label.bp3-label.bp3-inline .bp3-html-select,
      label.bp3-label.bp3-inline .bp3-input,
      label.bp3-label.bp3-inline .bp3-input-group,
      label.bp3-label.bp3-inline .bp3-select,
      label.bp3-label.bp3-inline .bp3-popover-wrapper{
        display:inline-block;
        margin:0 0 0 5px;
        vertical-align:top; }
      label.bp3-label.bp3-inline .bp3-button-group{
        margin:0 0 0 5px; }
      label.bp3-label.bp3-inline .bp3-input-group .bp3-input{
        margin-left:0; }
      label.bp3-label.bp3-inline.bp3-large{
        line-height:40px; }
    label.bp3-label:not(.bp3-inline) .bp3-popover-target{
      display:block; }
    .bp3-dark label.bp3-label{
      color:#f5f8fa; }
      .bp3-dark label.bp3-label.bp3-disabled,
      .bp3-dark label.bp3-label.bp3-disabled .bp3-text-muted{
        color:rgba(167, 182, 194, 0.6); }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button{
    -webkit-box-flex:1;
        -ms-flex:1 1 14px;
            flex:1 1 14px;
    min-height:0;
    padding:0;
    width:30px; }
    .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:first-child{
      border-radius:0 3px 0 0; }
    .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:last-child{
      border-radius:0 0 3px 0; }
  
  .bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:first-child{
    border-radius:3px 0 0 0; }
  
  .bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:last-child{
    border-radius:0 0 0 3px; }
  
  .bp3-numeric-input.bp3-large .bp3-button-group.bp3-vertical > .bp3-button{
    width:40px; }
  
  form{
    display:block; }
  .bp3-html-select select,
  .bp3-select select{
    display:-webkit-inline-box;
    display:-ms-inline-flexbox;
    display:inline-flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    border:none;
    border-radius:3px;
    cursor:pointer;
    font-size:14px;
    -webkit-box-pack:center;
        -ms-flex-pack:center;
            justify-content:center;
    padding:5px 10px;
    text-align:left;
    vertical-align:middle;
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026;
    -moz-appearance:none;
    -webkit-appearance:none;
    border-radius:3px;
    height:30px;
    padding:0 25px 0 10px;
    width:100%; }
    .bp3-html-select select > *, .bp3-select select > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-html-select select > .bp3-fill, .bp3-select select > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-html-select select::before,
    .bp3-select select::before, .bp3-html-select select > *, .bp3-select select > *{
      margin-right:7px; }
    .bp3-html-select select:empty::before,
    .bp3-select select:empty::before,
    .bp3-html-select select > :last-child,
    .bp3-select select > :last-child{
      margin-right:0; }
    .bp3-html-select select:hover,
    .bp3-select select:hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-html-select select:active,
    .bp3-select select:active, .bp3-html-select select.bp3-active,
    .bp3-select select.bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-html-select select:disabled,
    .bp3-select select:disabled, .bp3-html-select select.bp3-disabled,
    .bp3-select select.bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-html-select select:disabled.bp3-active,
      .bp3-select select:disabled.bp3-active, .bp3-html-select select:disabled.bp3-active:hover,
      .bp3-select select:disabled.bp3-active:hover, .bp3-html-select select.bp3-disabled.bp3-active,
      .bp3-select select.bp3-disabled.bp3-active, .bp3-html-select select.bp3-disabled.bp3-active:hover,
      .bp3-select select.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  
  .bp3-html-select.bp3-minimal select,
  .bp3-select.bp3-minimal select{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-html-select.bp3-minimal select:hover,
    .bp3-select.bp3-minimal select:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-html-select.bp3-minimal select:active,
    .bp3-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal select.bp3-active,
    .bp3-select.bp3-minimal select.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-html-select.bp3-minimal select:disabled,
    .bp3-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal select:disabled:hover,
    .bp3-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal select.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal select.bp3-disabled:hover,
    .bp3-select.bp3-minimal select.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-html-select.bp3-minimal select:disabled.bp3-active,
      .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active,
      .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active,
      .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-html-select.bp3-minimal select, .bp3-html-select.bp3-minimal .bp3-dark select,
    .bp3-dark .bp3-select.bp3-minimal select, .bp3-select.bp3-minimal .bp3-dark select{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
      .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover, .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
      .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
      .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
      .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-html-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal .bp3-dark select:disabled,
      .bp3-dark .bp3-select.bp3-minimal select:disabled, .bp3-select.bp3-minimal .bp3-dark select:disabled, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover,
      .bp3-dark .bp3-select.bp3-minimal select:disabled:hover, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-html-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary,
    .bp3-select.bp3-minimal select.bp3-intent-primary{
      color:#106ba3; }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
      .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
      .bp3-select.bp3-minimal select.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled,
      .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-success,
    .bp3-select.bp3-minimal select.bp3-intent-success{
      color:#0d8050; }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
      .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
      .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
      .bp3-select.bp3-minimal select.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
      .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled,
      .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-html-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning,
    .bp3-select.bp3-minimal select.bp3-intent-warning{
      color:#bf7326; }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
      .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
      .bp3-select.bp3-minimal select.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled,
      .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger,
    .bp3-select.bp3-minimal select.bp3-intent-danger{
      color:#c23030; }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
      .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
      .bp3-select.bp3-minimal select.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled,
      .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active,
        .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active,
          .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  
  .bp3-html-select.bp3-large select,
  .bp3-select.bp3-large select{
    font-size:16px;
    height:40px;
    padding-right:35px; }
  
  .bp3-dark .bp3-html-select select, .bp3-dark .bp3-select select{
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover, .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-html-select select:disabled, .bp3-dark .bp3-select select:disabled, .bp3-dark .bp3-html-select select.bp3-disabled, .bp3-dark .bp3-select select.bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-html-select select:disabled.bp3-active, .bp3-dark .bp3-select select:disabled.bp3-active, .bp3-dark .bp3-html-select select.bp3-disabled.bp3-active, .bp3-dark .bp3-select select.bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-html-select select .bp3-button-spinner .bp3-spinner-head, .bp3-dark .bp3-select select .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
  
  .bp3-html-select select:disabled,
  .bp3-select select:disabled{
    background-color:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  
  .bp3-html-select .bp3-icon,
  .bp3-select .bp3-icon, .bp3-select::after{
    color:#5c7080;
    pointer-events:none;
    position:absolute;
    right:7px;
    top:7px; }
    .bp3-html-select .bp3-disabled.bp3-icon,
    .bp3-select .bp3-disabled.bp3-icon, .bp3-disabled.bp3-select::after{
      color:rgba(92, 112, 128, 0.6); }
  .bp3-html-select,
  .bp3-select{
    display:inline-block;
    letter-spacing:normal;
    position:relative;
    vertical-align:middle; }
    .bp3-html-select select::-ms-expand,
    .bp3-select select::-ms-expand{
      display:none; }
    .bp3-html-select .bp3-icon,
    .bp3-select .bp3-icon{
      color:#5c7080; }
      .bp3-html-select .bp3-icon:hover,
      .bp3-select .bp3-icon:hover{
        color:#182026; }
      .bp3-dark .bp3-html-select .bp3-icon, .bp3-dark
      .bp3-select .bp3-icon{
        color:#a7b6c2; }
        .bp3-dark .bp3-html-select .bp3-icon:hover, .bp3-dark
        .bp3-select .bp3-icon:hover{
          color:#f5f8fa; }
    .bp3-html-select.bp3-large::after,
    .bp3-html-select.bp3-large .bp3-icon,
    .bp3-select.bp3-large::after,
    .bp3-select.bp3-large .bp3-icon{
      right:12px;
      top:12px; }
    .bp3-html-select.bp3-fill,
    .bp3-html-select.bp3-fill select,
    .bp3-select.bp3-fill,
    .bp3-select.bp3-fill select{
      width:100%; }
    .bp3-dark .bp3-html-select option, .bp3-dark
    .bp3-select option{
      background-color:#30404d;
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select option:disabled, .bp3-dark
    .bp3-select option:disabled{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-html-select::after, .bp3-dark
    .bp3-select::after{
      color:#a7b6c2; }
  
  .bp3-select::after{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    content:""; }
  .bp3-running-text table, table.bp3-html-table{
    border-spacing:0;
    font-size:14px; }
    .bp3-running-text table th, table.bp3-html-table th,
    .bp3-running-text table td,
    table.bp3-html-table td{
      padding:11px;
      text-align:left;
      vertical-align:top; }
    .bp3-running-text table th, table.bp3-html-table th{
      color:#182026;
      font-weight:600; }
    
    .bp3-running-text table td,
    table.bp3-html-table td{
      color:#182026; }
    .bp3-running-text table tbody tr:first-child th, table.bp3-html-table tbody tr:first-child th,
    .bp3-running-text table tbody tr:first-child td,
    table.bp3-html-table tbody tr:first-child td{
      -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
              box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
    .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
      color:#f5f8fa; }
    .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
      color:#f5f8fa; }
    .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
    .bp3-dark .bp3-running-text table tbody tr:first-child td,
    .bp3-running-text .bp3-dark table tbody tr:first-child td,
    .bp3-dark table.bp3-html-table tbody tr:first-child td{
      -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
  
  table.bp3-html-table.bp3-html-table-condensed th,
  table.bp3-html-table.bp3-html-table-condensed td, table.bp3-html-table.bp3-small th,
  table.bp3-html-table.bp3-small td{
    padding-bottom:6px;
    padding-top:6px; }
  
  table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
    background:rgba(191, 204, 214, 0.15); }
  
  table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }
  
  table.bp3-html-table.bp3-html-table-bordered tbody tr td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
    table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
      -webkit-box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15);
              box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15); }
  
  table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
    -webkit-box-shadow:none;
            box-shadow:none; }
    table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:not(:first-child){
      -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
              box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }
  
  table.bp3-html-table.bp3-interactive tbody tr:hover td{
    background-color:rgba(191, 204, 214, 0.3);
    cursor:pointer; }
  
  table.bp3-html-table.bp3-interactive tbody tr:active td{
    background-color:rgba(191, 204, 214, 0.4); }
  
  .bp3-dark table.bp3-html-table{ }
    .bp3-dark table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
      background:rgba(92, 112, 128, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
      -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td{
      -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
      .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
        -webkit-box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15);
                box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
      -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
      .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:first-child{
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:hover td{
      background-color:rgba(92, 112, 128, 0.3);
      cursor:pointer; }
    .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:active td{
      background-color:rgba(92, 112, 128, 0.4); }
  
  .bp3-key-combo{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center; }
    .bp3-key-combo > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-key-combo > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-key-combo::before,
    .bp3-key-combo > *{
      margin-right:5px; }
    .bp3-key-combo:empty::before,
    .bp3-key-combo > :last-child{
      margin-right:0; }
  
  .bp3-hotkey-dialog{
    padding-bottom:0;
    top:40px; }
    .bp3-hotkey-dialog .bp3-dialog-body{
      margin:0;
      padding:0; }
    .bp3-hotkey-dialog .bp3-hotkey-label{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1; }
  
  .bp3-hotkey-column{
    margin:auto;
    max-height:80vh;
    overflow-y:auto;
    padding:30px; }
    .bp3-hotkey-column .bp3-heading{
      margin-bottom:20px; }
      .bp3-hotkey-column .bp3-heading:not(:first-child){
        margin-top:40px; }
  
  .bp3-hotkey{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-pack:justify;
        -ms-flex-pack:justify;
            justify-content:space-between;
    margin-left:0;
    margin-right:0; }
    .bp3-hotkey:not(:last-child){
      margin-bottom:10px; }
  .bp3-icon{
    display:inline-block;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    vertical-align:text-bottom; }
    .bp3-icon:not(:empty)::before{
      content:"" !important;
      content:unset !important; }
    .bp3-icon > svg{
      display:block; }
      .bp3-icon > svg:not([fill]){
        fill:currentColor; }
  
  .bp3-icon.bp3-intent-primary, .bp3-icon-standard.bp3-intent-primary, .bp3-icon-large.bp3-intent-primary{
    color:#106ba3; }
    .bp3-dark .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-icon-large.bp3-intent-primary{
      color:#48aff0; }
  
  .bp3-icon.bp3-intent-success, .bp3-icon-standard.bp3-intent-success, .bp3-icon-large.bp3-intent-success{
    color:#0d8050; }
    .bp3-dark .bp3-icon.bp3-intent-success, .bp3-dark .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-icon-large.bp3-intent-success{
      color:#3dcc91; }
  
  .bp3-icon.bp3-intent-warning, .bp3-icon-standard.bp3-intent-warning, .bp3-icon-large.bp3-intent-warning{
    color:#bf7326; }
    .bp3-dark .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-icon-large.bp3-intent-warning{
      color:#ffb366; }
  
  .bp3-icon.bp3-intent-danger, .bp3-icon-standard.bp3-intent-danger, .bp3-icon-large.bp3-intent-danger{
    color:#c23030; }
    .bp3-dark .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-icon-large.bp3-intent-danger{
      color:#ff7373; }
  
  span.bp3-icon-standard{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    display:inline-block; }
  
  span.bp3-icon-large{
    font-family:"Icons20", sans-serif;
    font-size:20px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    display:inline-block; }
  
  span.bp3-icon:empty{
    font-family:"Icons20";
    font-size:inherit;
    font-style:normal;
    font-weight:400;
    line-height:1; }
    span.bp3-icon:empty::before{
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased; }
  
  .bp3-icon-add::before{
    content:""; }
  
  .bp3-icon-add-column-left::before{
    content:""; }
  
  .bp3-icon-add-column-right::before{
    content:""; }
  
  .bp3-icon-add-row-bottom::before{
    content:""; }
  
  .bp3-icon-add-row-top::before{
    content:""; }
  
  .bp3-icon-add-to-artifact::before{
    content:""; }
  
  .bp3-icon-add-to-folder::before{
    content:""; }
  
  .bp3-icon-airplane::before{
    content:""; }
  
  .bp3-icon-align-center::before{
    content:""; }
  
  .bp3-icon-align-justify::before{
    content:""; }
  
  .bp3-icon-align-left::before{
    content:""; }
  
  .bp3-icon-align-right::before{
    content:""; }
  
  .bp3-icon-alignment-bottom::before{
    content:""; }
  
  .bp3-icon-alignment-horizontal-center::before{
    content:""; }
  
  .bp3-icon-alignment-left::before{
    content:""; }
  
  .bp3-icon-alignment-right::before{
    content:""; }
  
  .bp3-icon-alignment-top::before{
    content:""; }
  
  .bp3-icon-alignment-vertical-center::before{
    content:""; }
  
  .bp3-icon-annotation::before{
    content:""; }
  
  .bp3-icon-application::before{
    content:""; }
  
  .bp3-icon-applications::before{
    content:""; }
  
  .bp3-icon-archive::before{
    content:""; }
  
  .bp3-icon-arrow-bottom-left::before{
    content:"↙"; }
  
  .bp3-icon-arrow-bottom-right::before{
    content:"↘"; }
  
  .bp3-icon-arrow-down::before{
    content:"↓"; }
  
  .bp3-icon-arrow-left::before{
    content:"←"; }
  
  .bp3-icon-arrow-right::before{
    content:"→"; }
  
  .bp3-icon-arrow-top-left::before{
    content:"↖"; }
  
  .bp3-icon-arrow-top-right::before{
    content:"↗"; }
  
  .bp3-icon-arrow-up::before{
    content:"↑"; }
  
  .bp3-icon-arrows-horizontal::before{
    content:"↔"; }
  
  .bp3-icon-arrows-vertical::before{
    content:"↕"; }
  
  .bp3-icon-asterisk::before{
    content:"*"; }
  
  .bp3-icon-automatic-updates::before{
    content:""; }
  
  .bp3-icon-badge::before{
    content:""; }
  
  .bp3-icon-ban-circle::before{
    content:""; }
  
  .bp3-icon-bank-account::before{
    content:""; }
  
  .bp3-icon-barcode::before{
    content:""; }
  
  .bp3-icon-blank::before{
    content:""; }
  
  .bp3-icon-blocked-person::before{
    content:""; }
  
  .bp3-icon-bold::before{
    content:""; }
  
  .bp3-icon-book::before{
    content:""; }
  
  .bp3-icon-bookmark::before{
    content:""; }
  
  .bp3-icon-box::before{
    content:""; }
  
  .bp3-icon-briefcase::before{
    content:""; }
  
  .bp3-icon-bring-data::before{
    content:""; }
  
  .bp3-icon-build::before{
    content:""; }
  
  .bp3-icon-calculator::before{
    content:""; }
  
  .bp3-icon-calendar::before{
    content:""; }
  
  .bp3-icon-camera::before{
    content:""; }
  
  .bp3-icon-caret-down::before{
    content:"⌄"; }
  
  .bp3-icon-caret-left::before{
    content:"〈"; }
  
  .bp3-icon-caret-right::before{
    content:"〉"; }
  
  .bp3-icon-caret-up::before{
    content:"⌃"; }
  
  .bp3-icon-cell-tower::before{
    content:""; }
  
  .bp3-icon-changes::before{
    content:""; }
  
  .bp3-icon-chart::before{
    content:""; }
  
  .bp3-icon-chat::before{
    content:""; }
  
  .bp3-icon-chevron-backward::before{
    content:""; }
  
  .bp3-icon-chevron-down::before{
    content:""; }
  
  .bp3-icon-chevron-forward::before{
    content:""; }
  
  .bp3-icon-chevron-left::before{
    content:""; }
  
  .bp3-icon-chevron-right::before{
    content:""; }
  
  .bp3-icon-chevron-up::before{
    content:""; }
  
  .bp3-icon-circle::before{
    content:""; }
  
  .bp3-icon-circle-arrow-down::before{
    content:""; }
  
  .bp3-icon-circle-arrow-left::before{
    content:""; }
  
  .bp3-icon-circle-arrow-right::before{
    content:""; }
  
  .bp3-icon-circle-arrow-up::before{
    content:""; }
  
  .bp3-icon-citation::before{
    content:""; }
  
  .bp3-icon-clean::before{
    content:""; }
  
  .bp3-icon-clipboard::before{
    content:""; }
  
  .bp3-icon-cloud::before{
    content:"☁"; }
  
  .bp3-icon-cloud-download::before{
    content:""; }
  
  .bp3-icon-cloud-upload::before{
    content:""; }
  
  .bp3-icon-code::before{
    content:""; }
  
  .bp3-icon-code-block::before{
    content:""; }
  
  .bp3-icon-cog::before{
    content:""; }
  
  .bp3-icon-collapse-all::before{
    content:""; }
  
  .bp3-icon-column-layout::before{
    content:""; }
  
  .bp3-icon-comment::before{
    content:""; }
  
  .bp3-icon-comparison::before{
    content:""; }
  
  .bp3-icon-compass::before{
    content:""; }
  
  .bp3-icon-compressed::before{
    content:""; }
  
  .bp3-icon-confirm::before{
    content:""; }
  
  .bp3-icon-console::before{
    content:""; }
  
  .bp3-icon-contrast::before{
    content:""; }
  
  .bp3-icon-control::before{
    content:""; }
  
  .bp3-icon-credit-card::before{
    content:""; }
  
  .bp3-icon-cross::before{
    content:"✗"; }
  
  .bp3-icon-crown::before{
    content:""; }
  
  .bp3-icon-cube::before{
    content:""; }
  
  .bp3-icon-cube-add::before{
    content:""; }
  
  .bp3-icon-cube-remove::before{
    content:""; }
  
  .bp3-icon-curved-range-chart::before{
    content:""; }
  
  .bp3-icon-cut::before{
    content:""; }
  
  .bp3-icon-dashboard::before{
    content:""; }
  
  .bp3-icon-data-lineage::before{
    content:""; }
  
  .bp3-icon-database::before{
    content:""; }
  
  .bp3-icon-delete::before{
    content:""; }
  
  .bp3-icon-delta::before{
    content:"Δ"; }
  
  .bp3-icon-derive-column::before{
    content:""; }
  
  .bp3-icon-desktop::before{
    content:""; }
  
  .bp3-icon-diagnosis::before{
    content:""; }
  
  .bp3-icon-diagram-tree::before{
    content:""; }
  
  .bp3-icon-direction-left::before{
    content:""; }
  
  .bp3-icon-direction-right::before{
    content:""; }
  
  .bp3-icon-disable::before{
    content:""; }
  
  .bp3-icon-document::before{
    content:""; }
  
  .bp3-icon-document-open::before{
    content:""; }
  
  .bp3-icon-document-share::before{
    content:""; }
  
  .bp3-icon-dollar::before{
    content:"$"; }
  
  .bp3-icon-dot::before{
    content:"•"; }
  
  .bp3-icon-double-caret-horizontal::before{
    content:""; }
  
  .bp3-icon-double-caret-vertical::before{
    content:""; }
  
  .bp3-icon-double-chevron-down::before{
    content:""; }
  
  .bp3-icon-double-chevron-left::before{
    content:""; }
  
  .bp3-icon-double-chevron-right::before{
    content:""; }
  
  .bp3-icon-double-chevron-up::before{
    content:""; }
  
  .bp3-icon-doughnut-chart::before{
    content:""; }
  
  .bp3-icon-download::before{
    content:""; }
  
  .bp3-icon-drag-handle-horizontal::before{
    content:""; }
  
  .bp3-icon-drag-handle-vertical::before{
    content:""; }
  
  .bp3-icon-draw::before{
    content:""; }
  
  .bp3-icon-drive-time::before{
    content:""; }
  
  .bp3-icon-duplicate::before{
    content:""; }
  
  .bp3-icon-edit::before{
    content:"✎"; }
  
  .bp3-icon-eject::before{
    content:"⏏"; }
  
  .bp3-icon-endorsed::before{
    content:""; }
  
  .bp3-icon-envelope::before{
    content:"✉"; }
  
  .bp3-icon-equals::before{
    content:""; }
  
  .bp3-icon-eraser::before{
    content:""; }
  
  .bp3-icon-error::before{
    content:""; }
  
  .bp3-icon-euro::before{
    content:"€"; }
  
  .bp3-icon-exchange::before{
    content:""; }
  
  .bp3-icon-exclude-row::before{
    content:""; }
  
  .bp3-icon-expand-all::before{
    content:""; }
  
  .bp3-icon-export::before{
    content:""; }
  
  .bp3-icon-eye-off::before{
    content:""; }
  
  .bp3-icon-eye-on::before{
    content:""; }
  
  .bp3-icon-eye-open::before{
    content:""; }
  
  .bp3-icon-fast-backward::before{
    content:""; }
  
  .bp3-icon-fast-forward::before{
    content:""; }
  
  .bp3-icon-feed::before{
    content:""; }
  
  .bp3-icon-feed-subscribed::before{
    content:""; }
  
  .bp3-icon-film::before{
    content:""; }
  
  .bp3-icon-filter::before{
    content:""; }
  
  .bp3-icon-filter-keep::before{
    content:""; }
  
  .bp3-icon-filter-list::before{
    content:""; }
  
  .bp3-icon-filter-open::before{
    content:""; }
  
  .bp3-icon-filter-remove::before{
    content:""; }
  
  .bp3-icon-flag::before{
    content:"⚑"; }
  
  .bp3-icon-flame::before{
    content:""; }
  
  .bp3-icon-flash::before{
    content:""; }
  
  .bp3-icon-floppy-disk::before{
    content:""; }
  
  .bp3-icon-flow-branch::before{
    content:""; }
  
  .bp3-icon-flow-end::before{
    content:""; }
  
  .bp3-icon-flow-linear::before{
    content:""; }
  
  .bp3-icon-flow-review::before{
    content:""; }
  
  .bp3-icon-flow-review-branch::before{
    content:""; }
  
  .bp3-icon-flows::before{
    content:""; }
  
  .bp3-icon-folder-close::before{
    content:""; }
  
  .bp3-icon-folder-new::before{
    content:""; }
  
  .bp3-icon-folder-open::before{
    content:""; }
  
  .bp3-icon-folder-shared::before{
    content:""; }
  
  .bp3-icon-folder-shared-open::before{
    content:""; }
  
  .bp3-icon-follower::before{
    content:""; }
  
  .bp3-icon-following::before{
    content:""; }
  
  .bp3-icon-font::before{
    content:""; }
  
  .bp3-icon-fork::before{
    content:""; }
  
  .bp3-icon-form::before{
    content:""; }
  
  .bp3-icon-full-circle::before{
    content:""; }
  
  .bp3-icon-full-stacked-chart::before{
    content:""; }
  
  .bp3-icon-fullscreen::before{
    content:""; }
  
  .bp3-icon-function::before{
    content:""; }
  
  .bp3-icon-gantt-chart::before{
    content:""; }
  
  .bp3-icon-geolocation::before{
    content:""; }
  
  .bp3-icon-geosearch::before{
    content:""; }
  
  .bp3-icon-git-branch::before{
    content:""; }
  
  .bp3-icon-git-commit::before{
    content:""; }
  
  .bp3-icon-git-merge::before{
    content:""; }
  
  .bp3-icon-git-new-branch::before{
    content:""; }
  
  .bp3-icon-git-pull::before{
    content:""; }
  
  .bp3-icon-git-push::before{
    content:""; }
  
  .bp3-icon-git-repo::before{
    content:""; }
  
  .bp3-icon-glass::before{
    content:""; }
  
  .bp3-icon-globe::before{
    content:""; }
  
  .bp3-icon-globe-network::before{
    content:""; }
  
  .bp3-icon-graph::before{
    content:""; }
  
  .bp3-icon-graph-remove::before{
    content:""; }
  
  .bp3-icon-greater-than::before{
    content:""; }
  
  .bp3-icon-greater-than-or-equal-to::before{
    content:""; }
  
  .bp3-icon-grid::before{
    content:""; }
  
  .bp3-icon-grid-view::before{
    content:""; }
  
  .bp3-icon-group-objects::before{
    content:""; }
  
  .bp3-icon-grouped-bar-chart::before{
    content:""; }
  
  .bp3-icon-hand::before{
    content:""; }
  
  .bp3-icon-hand-down::before{
    content:""; }
  
  .bp3-icon-hand-left::before{
    content:""; }
  
  .bp3-icon-hand-right::before{
    content:""; }
  
  .bp3-icon-hand-up::before{
    content:""; }
  
  .bp3-icon-header::before{
    content:""; }
  
  .bp3-icon-header-one::before{
    content:""; }
  
  .bp3-icon-header-two::before{
    content:""; }
  
  .bp3-icon-headset::before{
    content:""; }
  
  .bp3-icon-heart::before{
    content:"♥"; }
  
  .bp3-icon-heart-broken::before{
    content:""; }
  
  .bp3-icon-heat-grid::before{
    content:""; }
  
  .bp3-icon-heatmap::before{
    content:""; }
  
  .bp3-icon-help::before{
    content:"?"; }
  
  .bp3-icon-helper-management::before{
    content:""; }
  
  .bp3-icon-highlight::before{
    content:""; }
  
  .bp3-icon-history::before{
    content:""; }
  
  .bp3-icon-home::before{
    content:"⌂"; }
  
  .bp3-icon-horizontal-bar-chart::before{
    content:""; }
  
  .bp3-icon-horizontal-bar-chart-asc::before{
    content:""; }
  
  .bp3-icon-horizontal-bar-chart-desc::before{
    content:""; }
  
  .bp3-icon-horizontal-distribution::before{
    content:""; }
  
  .bp3-icon-id-number::before{
    content:""; }
  
  .bp3-icon-image-rotate-left::before{
    content:""; }
  
  .bp3-icon-image-rotate-right::before{
    content:""; }
  
  .bp3-icon-import::before{
    content:""; }
  
  .bp3-icon-inbox::before{
    content:""; }
  
  .bp3-icon-inbox-filtered::before{
    content:""; }
  
  .bp3-icon-inbox-geo::before{
    content:""; }
  
  .bp3-icon-inbox-search::before{
    content:""; }
  
  .bp3-icon-inbox-update::before{
    content:""; }
  
  .bp3-icon-info-sign::before{
    content:"ℹ"; }
  
  .bp3-icon-inheritance::before{
    content:""; }
  
  .bp3-icon-inner-join::before{
    content:""; }
  
  .bp3-icon-insert::before{
    content:""; }
  
  .bp3-icon-intersection::before{
    content:""; }
  
  .bp3-icon-ip-address::before{
    content:""; }
  
  .bp3-icon-issue::before{
    content:""; }
  
  .bp3-icon-issue-closed::before{
    content:""; }
  
  .bp3-icon-issue-new::before{
    content:""; }
  
  .bp3-icon-italic::before{
    content:""; }
  
  .bp3-icon-join-table::before{
    content:""; }
  
  .bp3-icon-key::before{
    content:""; }
  
  .bp3-icon-key-backspace::before{
    content:""; }
  
  .bp3-icon-key-command::before{
    content:""; }
  
  .bp3-icon-key-control::before{
    content:""; }
  
  .bp3-icon-key-delete::before{
    content:""; }
  
  .bp3-icon-key-enter::before{
    content:""; }
  
  .bp3-icon-key-escape::before{
    content:""; }
  
  .bp3-icon-key-option::before{
    content:""; }
  
  .bp3-icon-key-shift::before{
    content:""; }
  
  .bp3-icon-key-tab::before{
    content:""; }
  
  .bp3-icon-known-vehicle::before{
    content:""; }
  
  .bp3-icon-lab-test::before{
    content:""; }
  
  .bp3-icon-label::before{
    content:""; }
  
  .bp3-icon-layer::before{
    content:""; }
  
  .bp3-icon-layers::before{
    content:""; }
  
  .bp3-icon-layout::before{
    content:""; }
  
  .bp3-icon-layout-auto::before{
    content:""; }
  
  .bp3-icon-layout-balloon::before{
    content:""; }
  
  .bp3-icon-layout-circle::before{
    content:""; }
  
  .bp3-icon-layout-grid::before{
    content:""; }
  
  .bp3-icon-layout-group-by::before{
    content:""; }
  
  .bp3-icon-layout-hierarchy::before{
    content:""; }
  
  .bp3-icon-layout-linear::before{
    content:""; }
  
  .bp3-icon-layout-skew-grid::before{
    content:""; }
  
  .bp3-icon-layout-sorted-clusters::before{
    content:""; }
  
  .bp3-icon-learning::before{
    content:""; }
  
  .bp3-icon-left-join::before{
    content:""; }
  
  .bp3-icon-less-than::before{
    content:""; }
  
  .bp3-icon-less-than-or-equal-to::before{
    content:""; }
  
  .bp3-icon-lifesaver::before{
    content:""; }
  
  .bp3-icon-lightbulb::before{
    content:""; }
  
  .bp3-icon-link::before{
    content:""; }
  
  .bp3-icon-list::before{
    content:"☰"; }
  
  .bp3-icon-list-columns::before{
    content:""; }
  
  .bp3-icon-list-detail-view::before{
    content:""; }
  
  .bp3-icon-locate::before{
    content:""; }
  
  .bp3-icon-lock::before{
    content:""; }
  
  .bp3-icon-log-in::before{
    content:""; }
  
  .bp3-icon-log-out::before{
    content:""; }
  
  .bp3-icon-manual::before{
    content:""; }
  
  .bp3-icon-manually-entered-data::before{
    content:""; }
  
  .bp3-icon-map::before{
    content:""; }
  
  .bp3-icon-map-create::before{
    content:""; }
  
  .bp3-icon-map-marker::before{
    content:""; }
  
  .bp3-icon-maximize::before{
    content:""; }
  
  .bp3-icon-media::before{
    content:""; }
  
  .bp3-icon-menu::before{
    content:""; }
  
  .bp3-icon-menu-closed::before{
    content:""; }
  
  .bp3-icon-menu-open::before{
    content:""; }
  
  .bp3-icon-merge-columns::before{
    content:""; }
  
  .bp3-icon-merge-links::before{
    content:""; }
  
  .bp3-icon-minimize::before{
    content:""; }
  
  .bp3-icon-minus::before{
    content:"−"; }
  
  .bp3-icon-mobile-phone::before{
    content:""; }
  
  .bp3-icon-mobile-video::before{
    content:""; }
  
  .bp3-icon-moon::before{
    content:""; }
  
  .bp3-icon-more::before{
    content:""; }
  
  .bp3-icon-mountain::before{
    content:""; }
  
  .bp3-icon-move::before{
    content:""; }
  
  .bp3-icon-mugshot::before{
    content:""; }
  
  .bp3-icon-multi-select::before{
    content:""; }
  
  .bp3-icon-music::before{
    content:""; }
  
  .bp3-icon-new-drawing::before{
    content:""; }
  
  .bp3-icon-new-grid-item::before{
    content:""; }
  
  .bp3-icon-new-layer::before{
    content:""; }
  
  .bp3-icon-new-layers::before{
    content:""; }
  
  .bp3-icon-new-link::before{
    content:""; }
  
  .bp3-icon-new-object::before{
    content:""; }
  
  .bp3-icon-new-person::before{
    content:""; }
  
  .bp3-icon-new-prescription::before{
    content:""; }
  
  .bp3-icon-new-text-box::before{
    content:""; }
  
  .bp3-icon-ninja::before{
    content:""; }
  
  .bp3-icon-not-equal-to::before{
    content:""; }
  
  .bp3-icon-notifications::before{
    content:""; }
  
  .bp3-icon-notifications-updated::before{
    content:""; }
  
  .bp3-icon-numbered-list::before{
    content:""; }
  
  .bp3-icon-numerical::before{
    content:""; }
  
  .bp3-icon-office::before{
    content:""; }
  
  .bp3-icon-offline::before{
    content:""; }
  
  .bp3-icon-oil-field::before{
    content:""; }
  
  .bp3-icon-one-column::before{
    content:""; }
  
  .bp3-icon-outdated::before{
    content:""; }
  
  .bp3-icon-page-layout::before{
    content:""; }
  
  .bp3-icon-panel-stats::before{
    content:""; }
  
  .bp3-icon-panel-table::before{
    content:""; }
  
  .bp3-icon-paperclip::before{
    content:""; }
  
  .bp3-icon-paragraph::before{
    content:""; }
  
  .bp3-icon-path::before{
    content:""; }
  
  .bp3-icon-path-search::before{
    content:""; }
  
  .bp3-icon-pause::before{
    content:""; }
  
  .bp3-icon-people::before{
    content:""; }
  
  .bp3-icon-percentage::before{
    content:""; }
  
  .bp3-icon-person::before{
    content:""; }
  
  .bp3-icon-phone::before{
    content:"☎"; }
  
  .bp3-icon-pie-chart::before{
    content:""; }
  
  .bp3-icon-pin::before{
    content:""; }
  
  .bp3-icon-pivot::before{
    content:""; }
  
  .bp3-icon-pivot-table::before{
    content:""; }
  
  .bp3-icon-play::before{
    content:""; }
  
  .bp3-icon-plus::before{
    content:"+"; }
  
  .bp3-icon-polygon-filter::before{
    content:""; }
  
  .bp3-icon-power::before{
    content:""; }
  
  .bp3-icon-predictive-analysis::before{
    content:""; }
  
  .bp3-icon-prescription::before{
    content:""; }
  
  .bp3-icon-presentation::before{
    content:""; }
  
  .bp3-icon-print::before{
    content:"⎙"; }
  
  .bp3-icon-projects::before{
    content:""; }
  
  .bp3-icon-properties::before{
    content:""; }
  
  .bp3-icon-property::before{
    content:""; }
  
  .bp3-icon-publish-function::before{
    content:""; }
  
  .bp3-icon-pulse::before{
    content:""; }
  
  .bp3-icon-random::before{
    content:""; }
  
  .bp3-icon-record::before{
    content:""; }
  
  .bp3-icon-redo::before{
    content:""; }
  
  .bp3-icon-refresh::before{
    content:""; }
  
  .bp3-icon-regression-chart::before{
    content:""; }
  
  .bp3-icon-remove::before{
    content:""; }
  
  .bp3-icon-remove-column::before{
    content:""; }
  
  .bp3-icon-remove-column-left::before{
    content:""; }
  
  .bp3-icon-remove-column-right::before{
    content:""; }
  
  .bp3-icon-remove-row-bottom::before{
    content:""; }
  
  .bp3-icon-remove-row-top::before{
    content:""; }
  
  .bp3-icon-repeat::before{
    content:""; }
  
  .bp3-icon-reset::before{
    content:""; }
  
  .bp3-icon-resolve::before{
    content:""; }
  
  .bp3-icon-rig::before{
    content:""; }
  
  .bp3-icon-right-join::before{
    content:""; }
  
  .bp3-icon-ring::before{
    content:""; }
  
  .bp3-icon-rotate-document::before{
    content:""; }
  
  .bp3-icon-rotate-page::before{
    content:""; }
  
  .bp3-icon-satellite::before{
    content:""; }
  
  .bp3-icon-saved::before{
    content:""; }
  
  .bp3-icon-scatter-plot::before{
    content:""; }
  
  .bp3-icon-search::before{
    content:""; }
  
  .bp3-icon-search-around::before{
    content:""; }
  
  .bp3-icon-search-template::before{
    content:""; }
  
  .bp3-icon-search-text::before{
    content:""; }
  
  .bp3-icon-segmented-control::before{
    content:""; }
  
  .bp3-icon-select::before{
    content:""; }
  
  .bp3-icon-selection::before{
    content:"⦿"; }
  
  .bp3-icon-send-to::before{
    content:""; }
  
  .bp3-icon-send-to-graph::before{
    content:""; }
  
  .bp3-icon-send-to-map::before{
    content:""; }
  
  .bp3-icon-series-add::before{
    content:""; }
  
  .bp3-icon-series-configuration::before{
    content:""; }
  
  .bp3-icon-series-derived::before{
    content:""; }
  
  .bp3-icon-series-filtered::before{
    content:""; }
  
  .bp3-icon-series-search::before{
    content:""; }
  
  .bp3-icon-settings::before{
    content:""; }
  
  .bp3-icon-share::before{
    content:""; }
  
  .bp3-icon-shield::before{
    content:""; }
  
  .bp3-icon-shop::before{
    content:""; }
  
  .bp3-icon-shopping-cart::before{
    content:""; }
  
  .bp3-icon-signal-search::before{
    content:""; }
  
  .bp3-icon-sim-card::before{
    content:""; }
  
  .bp3-icon-slash::before{
    content:""; }
  
  .bp3-icon-small-cross::before{
    content:""; }
  
  .bp3-icon-small-minus::before{
    content:""; }
  
  .bp3-icon-small-plus::before{
    content:""; }
  
  .bp3-icon-small-tick::before{
    content:""; }
  
  .bp3-icon-snowflake::before{
    content:""; }
  
  .bp3-icon-social-media::before{
    content:""; }
  
  .bp3-icon-sort::before{
    content:""; }
  
  .bp3-icon-sort-alphabetical::before{
    content:""; }
  
  .bp3-icon-sort-alphabetical-desc::before{
    content:""; }
  
  .bp3-icon-sort-asc::before{
    content:""; }
  
  .bp3-icon-sort-desc::before{
    content:""; }
  
  .bp3-icon-sort-numerical::before{
    content:""; }
  
  .bp3-icon-sort-numerical-desc::before{
    content:""; }
  
  .bp3-icon-split-columns::before{
    content:""; }
  
  .bp3-icon-square::before{
    content:""; }
  
  .bp3-icon-stacked-chart::before{
    content:""; }
  
  .bp3-icon-star::before{
    content:"★"; }
  
  .bp3-icon-star-empty::before{
    content:"☆"; }
  
  .bp3-icon-step-backward::before{
    content:""; }
  
  .bp3-icon-step-chart::before{
    content:""; }
  
  .bp3-icon-step-forward::before{
    content:""; }
  
  .bp3-icon-stop::before{
    content:""; }
  
  .bp3-icon-stopwatch::before{
    content:""; }
  
  .bp3-icon-strikethrough::before{
    content:""; }
  
  .bp3-icon-style::before{
    content:""; }
  
  .bp3-icon-swap-horizontal::before{
    content:""; }
  
  .bp3-icon-swap-vertical::before{
    content:""; }
  
  .bp3-icon-symbol-circle::before{
    content:""; }
  
  .bp3-icon-symbol-cross::before{
    content:""; }
  
  .bp3-icon-symbol-diamond::before{
    content:""; }
  
  .bp3-icon-symbol-square::before{
    content:""; }
  
  .bp3-icon-symbol-triangle-down::before{
    content:""; }
  
  .bp3-icon-symbol-triangle-up::before{
    content:""; }
  
  .bp3-icon-tag::before{
    content:""; }
  
  .bp3-icon-take-action::before{
    content:""; }
  
  .bp3-icon-taxi::before{
    content:""; }
  
  .bp3-icon-text-highlight::before{
    content:""; }
  
  .bp3-icon-th::before{
    content:""; }
  
  .bp3-icon-th-derived::before{
    content:""; }
  
  .bp3-icon-th-disconnect::before{
    content:""; }
  
  .bp3-icon-th-filtered::before{
    content:""; }
  
  .bp3-icon-th-list::before{
    content:""; }
  
  .bp3-icon-thumbs-down::before{
    content:""; }
  
  .bp3-icon-thumbs-up::before{
    content:""; }
  
  .bp3-icon-tick::before{
    content:"✓"; }
  
  .bp3-icon-tick-circle::before{
    content:""; }
  
  .bp3-icon-time::before{
    content:"⏲"; }
  
  .bp3-icon-timeline-area-chart::before{
    content:""; }
  
  .bp3-icon-timeline-bar-chart::before{
    content:""; }
  
  .bp3-icon-timeline-events::before{
    content:""; }
  
  .bp3-icon-timeline-line-chart::before{
    content:""; }
  
  .bp3-icon-tint::before{
    content:""; }
  
  .bp3-icon-torch::before{
    content:""; }
  
  .bp3-icon-tractor::before{
    content:""; }
  
  .bp3-icon-train::before{
    content:""; }
  
  .bp3-icon-translate::before{
    content:""; }
  
  .bp3-icon-trash::before{
    content:""; }
  
  .bp3-icon-tree::before{
    content:""; }
  
  .bp3-icon-trending-down::before{
    content:""; }
  
  .bp3-icon-trending-up::before{
    content:""; }
  
  .bp3-icon-truck::before{
    content:""; }
  
  .bp3-icon-two-columns::before{
    content:""; }
  
  .bp3-icon-unarchive::before{
    content:""; }
  
  .bp3-icon-underline::before{
    content:"⎁"; }
  
  .bp3-icon-undo::before{
    content:"⎌"; }
  
  .bp3-icon-ungroup-objects::before{
    content:""; }
  
  .bp3-icon-unknown-vehicle::before{
    content:""; }
  
  .bp3-icon-unlock::before{
    content:""; }
  
  .bp3-icon-unpin::before{
    content:""; }
  
  .bp3-icon-unresolve::before{
    content:""; }
  
  .bp3-icon-updated::before{
    content:""; }
  
  .bp3-icon-upload::before{
    content:""; }
  
  .bp3-icon-user::before{
    content:""; }
  
  .bp3-icon-variable::before{
    content:""; }
  
  .bp3-icon-vertical-bar-chart-asc::before{
    content:""; }
  
  .bp3-icon-vertical-bar-chart-desc::before{
    content:""; }
  
  .bp3-icon-vertical-distribution::before{
    content:""; }
  
  .bp3-icon-video::before{
    content:""; }
  
  .bp3-icon-volume-down::before{
    content:""; }
  
  .bp3-icon-volume-off::before{
    content:""; }
  
  .bp3-icon-volume-up::before{
    content:""; }
  
  .bp3-icon-walk::before{
    content:""; }
  
  .bp3-icon-warning-sign::before{
    content:""; }
  
  .bp3-icon-waterfall-chart::before{
    content:""; }
  
  .bp3-icon-widget::before{
    content:""; }
  
  .bp3-icon-widget-button::before{
    content:""; }
  
  .bp3-icon-widget-footer::before{
    content:""; }
  
  .bp3-icon-widget-header::before{
    content:""; }
  
  .bp3-icon-wrench::before{
    content:""; }
  
  .bp3-icon-zoom-in::before{
    content:""; }
  
  .bp3-icon-zoom-out::before{
    content:""; }
  
  .bp3-icon-zoom-to-fit::before{
    content:""; }
  .bp3-submenu > .bp3-popover-wrapper{
    display:block; }
  
  .bp3-submenu .bp3-popover-target{
    display:block; }
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{ }
  
  .bp3-submenu.bp3-popover{
    -webkit-box-shadow:none;
            box-shadow:none;
    padding:0 5px; }
    .bp3-submenu.bp3-popover > .bp3-popover-content{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-submenu.bp3-popover, .bp3-submenu.bp3-popover.bp3-dark{
      -webkit-box-shadow:none;
              box-shadow:none; }
      .bp3-dark .bp3-submenu.bp3-popover > .bp3-popover-content, .bp3-submenu.bp3-popover.bp3-dark > .bp3-popover-content{
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
  .bp3-menu{
    background:#ffffff;
    border-radius:3px;
    color:#182026;
    list-style:none;
    margin:0;
    min-width:180px;
    padding:5px;
    text-align:left; }
  
  .bp3-menu-divider{
    border-top:1px solid rgba(16, 22, 26, 0.15);
    display:block;
    margin:5px; }
    .bp3-dark .bp3-menu-divider{
      border-top-color:rgba(255, 255, 255, 0.15); }
  
  .bp3-menu-item{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    border-radius:2px;
    color:inherit;
    line-height:20px;
    padding:5px 7px;
    text-decoration:none;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-menu-item > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-menu-item > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-menu-item::before,
    .bp3-menu-item > *{
      margin-right:7px; }
    .bp3-menu-item:empty::before,
    .bp3-menu-item > :last-child{
      margin-right:0; }
    .bp3-menu-item > .bp3-fill{
      word-break:break-word; }
    .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      background-color:rgba(167, 182, 194, 0.3);
      cursor:pointer;
      text-decoration:none; }
    .bp3-menu-item.bp3-disabled{
      background-color:inherit;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
    .bp3-dark .bp3-menu-item{
      color:inherit; }
      .bp3-dark .bp3-menu-item:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
        background-color:rgba(138, 155, 168, 0.15);
        color:inherit; }
      .bp3-dark .bp3-menu-item.bp3-disabled{
        background-color:inherit;
        color:rgba(167, 182, 194, 0.6); }
    .bp3-menu-item.bp3-intent-primary{
      color:#106ba3; }
      .bp3-menu-item.bp3-intent-primary .bp3-icon{
        color:inherit; }
      .bp3-menu-item.bp3-intent-primary::before, .bp3-menu-item.bp3-intent-primary::after,
      .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
        color:#106ba3; }
      .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary.bp3-active{
        background-color:#137cbd; }
      .bp3-menu-item.bp3-intent-primary:active{
        background-color:#106ba3; }
      .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
      .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
      .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary:active, .bp3-menu-item.bp3-intent-primary:active::before, .bp3-menu-item.bp3-intent-primary:active::after,
      .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-menu-item.bp3-intent-primary.bp3-active::after,
      .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-menu-item.bp3-intent-success{
      color:#0d8050; }
      .bp3-menu-item.bp3-intent-success .bp3-icon{
        color:inherit; }
      .bp3-menu-item.bp3-intent-success::before, .bp3-menu-item.bp3-intent-success::after,
      .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
        color:#0d8050; }
      .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success.bp3-active{
        background-color:#0f9960; }
      .bp3-menu-item.bp3-intent-success:active{
        background-color:#0d8050; }
      .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-menu-item.bp3-intent-success:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
      .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
      .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success:active, .bp3-menu-item.bp3-intent-success:active::before, .bp3-menu-item.bp3-intent-success:active::after,
      .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-menu-item.bp3-intent-success.bp3-active::after,
      .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-menu-item.bp3-intent-warning{
      color:#bf7326; }
      .bp3-menu-item.bp3-intent-warning .bp3-icon{
        color:inherit; }
      .bp3-menu-item.bp3-intent-warning::before, .bp3-menu-item.bp3-intent-warning::after,
      .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
        color:#bf7326; }
      .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning.bp3-active{
        background-color:#d9822b; }
      .bp3-menu-item.bp3-intent-warning:active{
        background-color:#bf7326; }
      .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
      .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
      .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning:active, .bp3-menu-item.bp3-intent-warning:active::before, .bp3-menu-item.bp3-intent-warning:active::after,
      .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-menu-item.bp3-intent-warning.bp3-active::after,
      .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-menu-item.bp3-intent-danger{
      color:#c23030; }
      .bp3-menu-item.bp3-intent-danger .bp3-icon{
        color:inherit; }
      .bp3-menu-item.bp3-intent-danger::before, .bp3-menu-item.bp3-intent-danger::after,
      .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
        color:#c23030; }
      .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger.bp3-active{
        background-color:#db3737; }
      .bp3-menu-item.bp3-intent-danger:active{
        background-color:#c23030; }
      .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
      .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
      .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger:active, .bp3-menu-item.bp3-intent-danger:active::before, .bp3-menu-item.bp3-intent-danger:active::after,
      .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-menu-item.bp3-intent-danger.bp3-active::after,
      .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-menu-item::before{
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      margin-right:7px; }
    .bp3-menu-item::before,
    .bp3-menu-item > .bp3-icon{
      color:#5c7080;
      margin-top:2px; }
    .bp3-menu-item .bp3-menu-item-label{
      color:#5c7080; }
    .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      color:inherit; }
    .bp3-menu-item.bp3-active, .bp3-menu-item:active{
      background-color:rgba(115, 134, 148, 0.3); }
    .bp3-menu-item.bp3-disabled{
      background-color:inherit !important;
      color:rgba(92, 112, 128, 0.6) !important;
      cursor:not-allowed !important;
      outline:none !important; }
      .bp3-menu-item.bp3-disabled::before,
      .bp3-menu-item.bp3-disabled > .bp3-icon,
      .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
        color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-large .bp3-menu-item{
      font-size:16px;
      line-height:22px;
      padding:9px 7px; }
      .bp3-large .bp3-menu-item .bp3-icon{
        margin-top:3px; }
      .bp3-large .bp3-menu-item::before{
        font-family:"Icons20", sans-serif;
        font-size:20px;
        font-style:normal;
        font-weight:400;
        line-height:1;
        -moz-osx-font-smoothing:grayscale;
        -webkit-font-smoothing:antialiased;
        margin-right:10px;
        margin-top:1px; }
  
  button.bp3-menu-item{
    background:none;
    border:none;
    text-align:left;
    width:100%; }
  .bp3-menu-header{
    border-top:1px solid rgba(16, 22, 26, 0.15);
    display:block;
    margin:5px;
    cursor:default;
    padding-left:2px; }
    .bp3-dark .bp3-menu-header{
      border-top-color:rgba(255, 255, 255, 0.15); }
    .bp3-menu-header:first-of-type{
      border-top:none; }
    .bp3-menu-header > h6{
      color:#182026;
      font-weight:600;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      word-wrap:normal;
      line-height:17px;
      margin:0;
      padding:10px 7px 0 1px; }
      .bp3-dark .bp3-menu-header > h6{
        color:#f5f8fa; }
    .bp3-menu-header:first-of-type > h6{
      padding-top:0; }
    .bp3-large .bp3-menu-header > h6{
      font-size:18px;
      padding-bottom:5px;
      padding-top:15px; }
    .bp3-large .bp3-menu-header:first-of-type > h6{
      padding-top:0; }
  
  .bp3-dark .bp3-menu{
    background:#30404d;
    color:#f5f8fa; }
  
  .bp3-dark .bp3-menu-item{ }
    .bp3-dark .bp3-menu-item.bp3-intent-primary{
      color:#48aff0; }
      .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-icon{
        color:inherit; }
      .bp3-dark .bp3-menu-item.bp3-intent-primary::before, .bp3-dark .bp3-menu-item.bp3-intent-primary::after,
      .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
        color:#48aff0; }
      .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active{
        background-color:#137cbd; }
      .bp3-dark .bp3-menu-item.bp3-intent-primary:active{
        background-color:#106ba3; }
      .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
      .bp3-dark .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
      .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label,
      .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary:active, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-dark .bp3-menu-item.bp3-intent-success{
      color:#3dcc91; }
      .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-icon{
        color:inherit; }
      .bp3-dark .bp3-menu-item.bp3-intent-success::before, .bp3-dark .bp3-menu-item.bp3-intent-success::after,
      .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
        color:#3dcc91; }
      .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active{
        background-color:#0f9960; }
      .bp3-dark .bp3-menu-item.bp3-intent-success:active{
        background-color:#0d8050; }
      .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
      .bp3-dark .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
      .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label,
      .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success:active, .bp3-dark .bp3-menu-item.bp3-intent-success:active::before, .bp3-dark .bp3-menu-item.bp3-intent-success:active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning{
      color:#ffb366; }
      .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-icon{
        color:inherit; }
      .bp3-dark .bp3-menu-item.bp3-intent-warning::before, .bp3-dark .bp3-menu-item.bp3-intent-warning::after,
      .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
        color:#ffb366; }
      .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active{
        background-color:#d9822b; }
      .bp3-dark .bp3-menu-item.bp3-intent-warning:active{
        background-color:#bf7326; }
      .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
      .bp3-dark .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
      .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label,
      .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning:active, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger{
      color:#ff7373; }
      .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-icon{
        color:inherit; }
      .bp3-dark .bp3-menu-item.bp3-intent-danger::before, .bp3-dark .bp3-menu-item.bp3-intent-danger::after,
      .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
        color:#ff7373; }
      .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active{
        background-color:#db3737; }
      .bp3-dark .bp3-menu-item.bp3-intent-danger:active{
        background-color:#c23030; }
      .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
      .bp3-dark .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
      .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label,
      .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger:active, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::after,
      .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
        color:#ffffff; }
    .bp3-dark .bp3-menu-item::before,
    .bp3-dark .bp3-menu-item > .bp3-icon{
      color:#a7b6c2; }
    .bp3-dark .bp3-menu-item .bp3-menu-item-label{
      color:#a7b6c2; }
    .bp3-dark .bp3-menu-item.bp3-active, .bp3-dark .bp3-menu-item:active{
      background-color:rgba(138, 155, 168, 0.3); }
    .bp3-dark .bp3-menu-item.bp3-disabled{
      color:rgba(167, 182, 194, 0.6) !important; }
      .bp3-dark .bp3-menu-item.bp3-disabled::before,
      .bp3-dark .bp3-menu-item.bp3-disabled > .bp3-icon,
      .bp3-dark .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
        color:rgba(167, 182, 194, 0.6) !important; }
  
  .bp3-dark .bp3-menu-divider,
  .bp3-dark .bp3-menu-header{
    border-color:rgba(255, 255, 255, 0.15); }
  
  .bp3-dark .bp3-menu-header > h6{
    color:#f5f8fa; }
  
  .bp3-label .bp3-menu{
    margin-top:5px; }
  .bp3-navbar{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
    height:50px;
    padding:0 15px;
    position:relative;
    width:100%;
    z-index:10; }
    .bp3-navbar.bp3-dark,
    .bp3-dark .bp3-navbar{
      background-color:#394b59; }
    .bp3-navbar.bp3-dark{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-navbar{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-navbar.bp3-fixed-top{
      left:0;
      position:fixed;
      right:0;
      top:0; }
  
  .bp3-navbar-heading{
    font-size:16px;
    margin-right:15px; }
  
  .bp3-navbar-group{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    height:50px; }
    .bp3-navbar-group.bp3-align-left{
      float:left; }
    .bp3-navbar-group.bp3-align-right{
      float:right; }
  
  .bp3-navbar-divider{
    border-left:1px solid rgba(16, 22, 26, 0.15);
    height:20px;
    margin:0 10px; }
    .bp3-dark .bp3-navbar-divider{
      border-left-color:rgba(255, 255, 255, 0.15); }
  .bp3-non-ideal-state{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    height:100%;
    -webkit-box-pack:center;
        -ms-flex-pack:center;
            justify-content:center;
    text-align:center;
    width:100%; }
    .bp3-non-ideal-state > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-non-ideal-state > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-non-ideal-state::before,
    .bp3-non-ideal-state > *{
      margin-bottom:20px; }
    .bp3-non-ideal-state:empty::before,
    .bp3-non-ideal-state > :last-child{
      margin-bottom:0; }
    .bp3-non-ideal-state > *{
      max-width:400px; }
  
  .bp3-non-ideal-state-visual{
    color:rgba(92, 112, 128, 0.6);
    font-size:60px; }
    .bp3-dark .bp3-non-ideal-state-visual{
      color:rgba(167, 182, 194, 0.6); }
  
  .bp3-overflow-list{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -ms-flex-wrap:nowrap;
        flex-wrap:nowrap;
    min-width:0; }
  
  .bp3-overflow-list-spacer{
    -ms-flex-negative:1;
        flex-shrink:1;
    width:1px; }
  
  body.bp3-overlay-open{
    overflow:hidden; }
  
  .bp3-overlay{
    bottom:0;
    left:0;
    position:static;
    right:0;
    top:0;
    z-index:20; }
    .bp3-overlay:not(.bp3-overlay-open){
      pointer-events:none; }
    .bp3-overlay.bp3-overlay-container{
      overflow:hidden;
      position:fixed; }
      .bp3-overlay.bp3-overlay-container.bp3-overlay-inline{
        position:absolute; }
    .bp3-overlay.bp3-overlay-scroll-container{
      overflow:auto;
      position:fixed; }
      .bp3-overlay.bp3-overlay-scroll-container.bp3-overlay-inline{
        position:absolute; }
    .bp3-overlay.bp3-overlay-inline{
      display:inline;
      overflow:visible; }
  
  .bp3-overlay-content{
    position:fixed;
    z-index:20; }
    .bp3-overlay-inline .bp3-overlay-content,
    .bp3-overlay-scroll-container .bp3-overlay-content{
      position:absolute; }
  
  .bp3-overlay-backdrop{
    bottom:0;
    left:0;
    position:fixed;
    right:0;
    top:0;
    opacity:1;
    background-color:rgba(16, 22, 26, 0.7);
    overflow:auto;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none;
    z-index:20; }
    .bp3-overlay-backdrop.bp3-overlay-enter, .bp3-overlay-backdrop.bp3-overlay-appear{
      opacity:0; }
    .bp3-overlay-backdrop.bp3-overlay-enter-active, .bp3-overlay-backdrop.bp3-overlay-appear-active{
      opacity:1;
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:opacity;
      transition-property:opacity;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-overlay-backdrop.bp3-overlay-exit{
      opacity:1; }
    .bp3-overlay-backdrop.bp3-overlay-exit-active{
      opacity:0;
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:opacity;
      transition-property:opacity;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-overlay-backdrop:focus{
      outline:none; }
    .bp3-overlay-inline .bp3-overlay-backdrop{
      position:absolute; }
  .bp3-panel-stack{
    overflow:hidden;
    position:relative; }
  
  .bp3-panel-stack-header{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
            box-shadow:0 1px rgba(16, 22, 26, 0.15);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -ms-flex-negative:0;
        flex-shrink:0;
    height:30px;
    z-index:1; }
    .bp3-dark .bp3-panel-stack-header{
      -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
              box-shadow:0 1px rgba(255, 255, 255, 0.15); }
    .bp3-panel-stack-header > span{
      -webkit-box-align:stretch;
          -ms-flex-align:stretch;
              align-items:stretch;
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex;
      -webkit-box-flex:1;
          -ms-flex:1;
              flex:1; }
    .bp3-panel-stack-header .bp3-heading{
      margin:0 5px; }
  
  .bp3-button.bp3-panel-stack-header-back{
    margin-left:5px;
    padding-left:0;
    white-space:nowrap; }
    .bp3-button.bp3-panel-stack-header-back .bp3-icon{
      margin:0 2px; }
  
  .bp3-panel-stack-view{
    bottom:0;
    left:0;
    position:absolute;
    right:0;
    top:0;
    background-color:#ffffff;
    border-right:1px solid rgba(16, 22, 26, 0.15);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    margin-right:-1px;
    overflow-y:auto;
    z-index:1; }
    .bp3-dark .bp3-panel-stack-view{
      background-color:#30404d; }
    .bp3-panel-stack-view:nth-last-child(n + 4){
      display:none; }
  
  .bp3-panel-stack-push .bp3-panel-stack-enter, .bp3-panel-stack-push .bp3-panel-stack-appear{
    -webkit-transform:translateX(100%);
            transform:translateX(100%);
    opacity:0; }
  
  .bp3-panel-stack-push .bp3-panel-stack-enter-active, .bp3-panel-stack-push .bp3-panel-stack-appear-active{
    -webkit-transform:translate(0%);
            transform:translate(0%);
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:400ms;
            transition-duration:400ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:transform, opacity;
    transition-property:transform, opacity, -webkit-transform;
    -webkit-transition-timing-function:ease;
            transition-timing-function:ease; }
  
  .bp3-panel-stack-push .bp3-panel-stack-exit{
    -webkit-transform:translate(0%);
            transform:translate(0%);
    opacity:1; }
  
  .bp3-panel-stack-push .bp3-panel-stack-exit-active{
    -webkit-transform:translateX(-50%);
            transform:translateX(-50%);
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:400ms;
            transition-duration:400ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:transform, opacity;
    transition-property:transform, opacity, -webkit-transform;
    -webkit-transition-timing-function:ease;
            transition-timing-function:ease; }
  
  .bp3-panel-stack-pop .bp3-panel-stack-enter, .bp3-panel-stack-pop .bp3-panel-stack-appear{
    -webkit-transform:translateX(-50%);
            transform:translateX(-50%);
    opacity:0; }
  
  .bp3-panel-stack-pop .bp3-panel-stack-enter-active, .bp3-panel-stack-pop .bp3-panel-stack-appear-active{
    -webkit-transform:translate(0%);
            transform:translate(0%);
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:400ms;
            transition-duration:400ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:transform, opacity;
    transition-property:transform, opacity, -webkit-transform;
    -webkit-transition-timing-function:ease;
            transition-timing-function:ease; }
  
  .bp3-panel-stack-pop .bp3-panel-stack-exit{
    -webkit-transform:translate(0%);
            transform:translate(0%);
    opacity:1; }
  
  .bp3-panel-stack-pop .bp3-panel-stack-exit-active{
    -webkit-transform:translateX(100%);
            transform:translateX(100%);
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:400ms;
            transition-duration:400ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:transform, opacity;
    transition-property:transform, opacity, -webkit-transform;
    -webkit-transition-timing-function:ease;
            transition-timing-function:ease; }
  .bp3-popover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    -webkit-transform:scale(1);
            transform:scale(1);
    border-radius:3px;
    display:inline-block;
    z-index:20; }
    .bp3-popover .bp3-popover-arrow{
      height:30px;
      position:absolute;
      width:30px; }
      .bp3-popover .bp3-popover-arrow::before{
        height:20px;
        margin:5px;
        width:20px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover{
      margin-bottom:17px;
      margin-top:-17px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
        bottom:-11px; }
        .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow svg{
          -webkit-transform:rotate(-90deg);
                  transform:rotate(-90deg); }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover{
      margin-left:17px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
        left:-11px; }
        .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow svg{
          -webkit-transform:rotate(0);
                  transform:rotate(0); }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover{
      margin-top:17px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
        top:-11px; }
        .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow svg{
          -webkit-transform:rotate(90deg);
                  transform:rotate(90deg); }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover{
      margin-left:-17px;
      margin-right:17px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
        right:-11px; }
        .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow svg{
          -webkit-transform:rotate(180deg);
                  transform:rotate(180deg); }
    .bp3-tether-element-attached-middle > .bp3-popover > .bp3-popover-arrow{
      top:50%;
      -webkit-transform:translateY(-50%);
              transform:translateY(-50%); }
    .bp3-tether-element-attached-center > .bp3-popover > .bp3-popover-arrow{
      right:50%;
      -webkit-transform:translateX(50%);
              transform:translateX(50%); }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
      top:-0.3934px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
      right:-0.3934px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
      left:-0.3934px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
      bottom:-0.3934px; }
    .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-popover{
      -webkit-transform-origin:top left;
              transform-origin:top left; }
    .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-popover{
      -webkit-transform-origin:top center;
              transform-origin:top center; }
    .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-popover{
      -webkit-transform-origin:top right;
              transform-origin:top right; }
    .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-popover{
      -webkit-transform-origin:center left;
              transform-origin:center left; }
    .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-popover{
      -webkit-transform-origin:center center;
              transform-origin:center center; }
    .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-popover{
      -webkit-transform-origin:center right;
              transform-origin:center right; }
    .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-popover{
      -webkit-transform-origin:bottom left;
              transform-origin:bottom left; }
    .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-popover{
      -webkit-transform-origin:bottom center;
              transform-origin:bottom center; }
    .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-popover{
      -webkit-transform-origin:bottom right;
              transform-origin:bottom right; }
    .bp3-popover .bp3-popover-content{
      background:#ffffff;
      color:inherit; }
    .bp3-popover .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
    .bp3-popover .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.1; }
    .bp3-popover .bp3-popover-arrow-fill{
      fill:#ffffff; }
    .bp3-popover-enter > .bp3-popover, .bp3-popover-appear > .bp3-popover{
      -webkit-transform:scale(0.3);
              transform:scale(0.3); }
    .bp3-popover-enter-active > .bp3-popover, .bp3-popover-appear-active > .bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:300ms;
              transition-duration:300ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
              transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
    .bp3-popover-exit > .bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1); }
    .bp3-popover-exit-active > .bp3-popover{
      -webkit-transform:scale(0.3);
              transform:scale(0.3);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:300ms;
              transition-duration:300ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
              transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
    .bp3-popover .bp3-popover-content{
      border-radius:3px;
      position:relative; }
    .bp3-popover.bp3-popover-content-sizing .bp3-popover-content{
      max-width:350px;
      padding:20px; }
    .bp3-popover-target + .bp3-overlay .bp3-popover.bp3-popover-content-sizing{
      width:350px; }
    .bp3-popover.bp3-minimal{
      margin:0 !important; }
      .bp3-popover.bp3-minimal .bp3-popover-arrow{
        display:none; }
      .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
        .bp3-popover-enter > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear > .bp3-popover.bp3-minimal.bp3-popover{
          -webkit-transform:scale(1);
                  transform:scale(1); }
        .bp3-popover-enter-active > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear-active > .bp3-popover.bp3-minimal.bp3-popover{
          -webkit-transform:scale(1);
                  transform:scale(1);
          -webkit-transition-delay:0;
                  transition-delay:0;
          -webkit-transition-duration:100ms;
                  transition-duration:100ms;
          -webkit-transition-property:-webkit-transform;
          transition-property:-webkit-transform;
          transition-property:transform;
          transition-property:transform, -webkit-transform;
          -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                  transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
        .bp3-popover-exit > .bp3-popover.bp3-minimal.bp3-popover{
          -webkit-transform:scale(1);
                  transform:scale(1); }
        .bp3-popover-exit-active > .bp3-popover.bp3-minimal.bp3-popover{
          -webkit-transform:scale(1);
                  transform:scale(1);
          -webkit-transition-delay:0;
                  transition-delay:0;
          -webkit-transition-duration:100ms;
                  transition-duration:100ms;
          -webkit-transition-property:-webkit-transform;
          transition-property:-webkit-transform;
          transition-property:transform;
          transition-property:transform, -webkit-transform;
          -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                  transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-popover.bp3-dark,
    .bp3-dark .bp3-popover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
      .bp3-popover.bp3-dark .bp3-popover-content,
      .bp3-dark .bp3-popover .bp3-popover-content{
        background:#30404d;
        color:inherit; }
      .bp3-popover.bp3-dark .bp3-popover-arrow::before,
      .bp3-dark .bp3-popover .bp3-popover-arrow::before{
        -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
                box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
      .bp3-popover.bp3-dark .bp3-popover-arrow-border,
      .bp3-dark .bp3-popover .bp3-popover-arrow-border{
        fill:#10161a;
        fill-opacity:0.2; }
      .bp3-popover.bp3-dark .bp3-popover-arrow-fill,
      .bp3-dark .bp3-popover .bp3-popover-arrow-fill{
        fill:#30404d; }
  
  .bp3-popover-arrow::before{
    border-radius:2px;
    content:"";
    display:block;
    position:absolute;
    -webkit-transform:rotate(45deg);
            transform:rotate(45deg); }
  
  .bp3-tether-pinned .bp3-popover-arrow{
    display:none; }
  
  .bp3-popover-backdrop{
    background:rgba(255, 255, 255, 0); }
  
  .bp3-transition-container{
    opacity:1;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    z-index:20; }
    .bp3-transition-container.bp3-popover-enter, .bp3-transition-container.bp3-popover-appear{
      opacity:0; }
    .bp3-transition-container.bp3-popover-enter-active, .bp3-transition-container.bp3-popover-appear-active{
      opacity:1;
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:opacity;
      transition-property:opacity;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-transition-container.bp3-popover-exit{
      opacity:1; }
    .bp3-transition-container.bp3-popover-exit-active{
      opacity:0;
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:opacity;
      transition-property:opacity;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-transition-container:focus{
      outline:none; }
    .bp3-transition-container.bp3-popover-leave .bp3-popover-content{
      pointer-events:none; }
    .bp3-transition-container[data-x-out-of-boundaries]{
      display:none; }
  
  span.bp3-popover-target{
    display:inline-block; }
  
  .bp3-popover-wrapper.bp3-fill{
    width:100%; }
  
  .bp3-portal{
    left:0;
    position:absolute;
    right:0;
    top:0; }
  @-webkit-keyframes linear-progress-bar-stripes{
    from{
      background-position:0 0; }
    to{
      background-position:30px 0; } }
  @keyframes linear-progress-bar-stripes{
    from{
      background-position:0 0; }
    to{
      background-position:30px 0; } }
  
  .bp3-progress-bar{
    background:rgba(92, 112, 128, 0.2);
    border-radius:40px;
    display:block;
    height:8px;
    overflow:hidden;
    position:relative;
    width:100%; }
    .bp3-progress-bar .bp3-progress-meter{
      background:linear-gradient(-45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%);
      background-color:rgba(92, 112, 128, 0.8);
      background-size:30px 30px;
      border-radius:40px;
      height:100%;
      position:absolute;
      -webkit-transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
      width:100%; }
    .bp3-progress-bar:not(.bp3-no-animation):not(.bp3-no-stripes) .bp3-progress-meter{
      animation:linear-progress-bar-stripes 300ms linear infinite reverse; }
    .bp3-progress-bar.bp3-no-stripes .bp3-progress-meter{
      background-image:none; }
  
  .bp3-dark .bp3-progress-bar{
    background:rgba(16, 22, 26, 0.5); }
    .bp3-dark .bp3-progress-bar .bp3-progress-meter{
      background-color:#8a9ba8; }
  
  .bp3-progress-bar.bp3-intent-primary .bp3-progress-meter{
    background-color:#137cbd; }
  
  .bp3-progress-bar.bp3-intent-success .bp3-progress-meter{
    background-color:#0f9960; }
  
  .bp3-progress-bar.bp3-intent-warning .bp3-progress-meter{
    background-color:#d9822b; }
  
  .bp3-progress-bar.bp3-intent-danger .bp3-progress-meter{
    background-color:#db3737; }
  @-webkit-keyframes skeleton-glow{
    from{
      background:rgba(206, 217, 224, 0.2);
      border-color:rgba(206, 217, 224, 0.2); }
    to{
      background:rgba(92, 112, 128, 0.2);
      border-color:rgba(92, 112, 128, 0.2); } }
  @keyframes skeleton-glow{
    from{
      background:rgba(206, 217, 224, 0.2);
      border-color:rgba(206, 217, 224, 0.2); }
    to{
      background:rgba(92, 112, 128, 0.2);
      border-color:rgba(92, 112, 128, 0.2); } }
  .bp3-skeleton{
    -webkit-animation:1000ms linear infinite alternate skeleton-glow;
            animation:1000ms linear infinite alternate skeleton-glow;
    background:rgba(206, 217, 224, 0.2);
    background-clip:padding-box !important;
    border-color:rgba(206, 217, 224, 0.2) !important;
    border-radius:2px;
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    color:transparent !important;
    cursor:default;
    pointer-events:none;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-skeleton::before, .bp3-skeleton::after,
    .bp3-skeleton *{
      visibility:hidden !important; }
  .bp3-slider{
    height:40px;
    min-width:150px;
    width:100%;
    cursor:default;
    outline:none;
    position:relative;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-slider:hover{
      cursor:pointer; }
    .bp3-slider:active{
      cursor:-webkit-grabbing;
      cursor:grabbing; }
    .bp3-slider.bp3-disabled{
      cursor:not-allowed;
      opacity:0.5; }
    .bp3-slider.bp3-slider-unlabeled{
      height:16px; }
  
  .bp3-slider-track,
  .bp3-slider-progress{
    height:6px;
    left:0;
    right:0;
    top:5px;
    position:absolute; }
  
  .bp3-slider-track{
    border-radius:3px;
    overflow:hidden; }
  
  .bp3-slider-progress{
    background:rgba(92, 112, 128, 0.2); }
    .bp3-dark .bp3-slider-progress{
      background:rgba(16, 22, 26, 0.5); }
    .bp3-slider-progress.bp3-intent-primary{
      background-color:#137cbd; }
    .bp3-slider-progress.bp3-intent-success{
      background-color:#0f9960; }
    .bp3-slider-progress.bp3-intent-warning{
      background-color:#d9822b; }
    .bp3-slider-progress.bp3-intent-danger{
      background-color:#db3737; }
  
  .bp3-slider-handle{
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
    cursor:pointer;
    height:16px;
    left:0;
    position:absolute;
    top:0;
    width:16px; }
    .bp3-slider-handle:hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-slider-handle:active, .bp3-slider-handle.bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-slider-handle:disabled, .bp3-slider-handle.bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-slider-handle:disabled.bp3-active, .bp3-slider-handle:disabled.bp3-active:hover, .bp3-slider-handle.bp3-disabled.bp3-active, .bp3-slider-handle.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
    .bp3-slider-handle:focus{
      z-index:1; }
    .bp3-slider-handle:hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
      cursor:-webkit-grab;
      cursor:grab;
      z-index:2; }
    .bp3-slider-handle.bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
      cursor:-webkit-grabbing;
      cursor:grabbing; }
    .bp3-disabled .bp3-slider-handle{
      background:#bfccd6;
      -webkit-box-shadow:none;
              box-shadow:none;
      pointer-events:none; }
    .bp3-dark .bp3-slider-handle{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark .bp3-slider-handle:hover, .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-slider-handle:hover{
        background-color:#30404d;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
        background-color:#202b33;
        background-image:none;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-slider-handle:disabled, .bp3-dark .bp3-slider-handle.bp3-disabled{
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-slider-handle:disabled.bp3-active, .bp3-dark .bp3-slider-handle.bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-slider-handle .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
      .bp3-dark .bp3-slider-handle, .bp3-dark .bp3-slider-handle:hover{
        background-color:#394b59; }
      .bp3-dark .bp3-slider-handle.bp3-active{
        background-color:#293742; }
    .bp3-dark .bp3-disabled .bp3-slider-handle{
      background:#5c7080;
      border-color:#5c7080;
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-slider-handle .bp3-slider-label{
      background:#394b59;
      border-radius:3px;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
      color:#f5f8fa;
      margin-left:8px; }
      .bp3-dark .bp3-slider-handle .bp3-slider-label{
        background:#e1e8ed;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
        color:#394b59; }
      .bp3-disabled .bp3-slider-handle .bp3-slider-label{
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-slider-handle.bp3-start, .bp3-slider-handle.bp3-end{
      width:8px; }
    .bp3-slider-handle.bp3-start{
      border-bottom-right-radius:0;
      border-top-right-radius:0; }
    .bp3-slider-handle.bp3-end{
      border-bottom-left-radius:0;
      border-top-left-radius:0;
      margin-left:8px; }
      .bp3-slider-handle.bp3-end .bp3-slider-label{
        margin-left:0; }
  
  .bp3-slider-label{
    -webkit-transform:translate(-50%, 20px);
            transform:translate(-50%, 20px);
    display:inline-block;
    font-size:12px;
    line-height:1;
    padding:2px 5px;
    position:absolute;
    vertical-align:top; }
  
  .bp3-slider.bp3-vertical{
    height:150px;
    min-width:40px;
    width:40px; }
    .bp3-slider.bp3-vertical .bp3-slider-track,
    .bp3-slider.bp3-vertical .bp3-slider-progress{
      bottom:0;
      height:auto;
      left:5px;
      top:0;
      width:6px; }
    .bp3-slider.bp3-vertical .bp3-slider-progress{
      top:auto; }
    .bp3-slider.bp3-vertical .bp3-slider-label{
      -webkit-transform:translate(20px, 50%);
              transform:translate(20px, 50%); }
    .bp3-slider.bp3-vertical .bp3-slider-handle{
      top:auto; }
      .bp3-slider.bp3-vertical .bp3-slider-handle .bp3-slider-label{
        margin-left:0;
        margin-top:-8px; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end, .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
        height:8px;
        margin-left:0;
        width:16px; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
        border-bottom-right-radius:3px;
        border-top-left-radius:0; }
        .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start .bp3-slider-label{
          -webkit-transform:translate(20px);
                  transform:translate(20px); }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end{
        border-bottom-left-radius:0;
        border-bottom-right-radius:0;
        border-top-left-radius:3px;
        margin-bottom:8px; }
  
  @-webkit-keyframes pt-spinner-animation{
    from{
      -webkit-transform:rotate(0deg);
              transform:rotate(0deg); }
    to{
      -webkit-transform:rotate(360deg);
              transform:rotate(360deg); } }
  
  @keyframes pt-spinner-animation{
    from{
      -webkit-transform:rotate(0deg);
              transform:rotate(0deg); }
    to{
      -webkit-transform:rotate(360deg);
              transform:rotate(360deg); } }
  
  .bp3-spinner{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-pack:center;
        -ms-flex-pack:center;
            justify-content:center;
    overflow:visible;
    vertical-align:middle; }
    .bp3-spinner svg{
      display:block; }
    .bp3-spinner path{
      fill-opacity:0; }
    .bp3-spinner .bp3-spinner-head{
      stroke:rgba(92, 112, 128, 0.8);
      stroke-linecap:round;
      -webkit-transform-origin:center;
              transform-origin:center;
      -webkit-transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-spinner .bp3-spinner-track{
      stroke:rgba(92, 112, 128, 0.2); }
  
  .bp3-spinner-animation{
    -webkit-animation:pt-spinner-animation 500ms linear infinite;
            animation:pt-spinner-animation 500ms linear infinite; }
    .bp3-no-spin > .bp3-spinner-animation{
      -webkit-animation:none;
              animation:none; }
  
  .bp3-dark .bp3-spinner .bp3-spinner-head{
    stroke:#8a9ba8; }
  
  .bp3-dark .bp3-spinner .bp3-spinner-track{
    stroke:rgba(16, 22, 26, 0.5); }
  
  .bp3-spinner.bp3-intent-primary .bp3-spinner-head{
    stroke:#137cbd; }
  
  .bp3-spinner.bp3-intent-success .bp3-spinner-head{
    stroke:#0f9960; }
  
  .bp3-spinner.bp3-intent-warning .bp3-spinner-head{
    stroke:#d9822b; }
  
  .bp3-spinner.bp3-intent-danger .bp3-spinner-head{
    stroke:#db3737; }
  .bp3-tabs.bp3-vertical{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list{
      -webkit-box-align:start;
          -ms-flex-align:start;
              align-items:flex-start;
      -webkit-box-orient:vertical;
      -webkit-box-direction:normal;
          -ms-flex-direction:column;
              flex-direction:column; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab{
        border-radius:3px;
        padding:0 10px;
        width:100%; }
        .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab[aria-selected="true"]{
          background-color:rgba(19, 124, 189, 0.2);
          -webkit-box-shadow:none;
                  box-shadow:none; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab-indicator-wrapper .bp3-tab-indicator{
        background-color:rgba(19, 124, 189, 0.2);
        border-radius:3px;
        bottom:0;
        height:auto;
        left:0;
        right:0;
        top:0; }
    .bp3-tabs.bp3-vertical > .bp3-tab-panel{
      margin-top:0;
      padding-left:20px; }
  
  .bp3-tab-list{
    -webkit-box-align:end;
        -ms-flex-align:end;
            align-items:flex-end;
    border:none;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    list-style:none;
    margin:0;
    padding:0;
    position:relative; }
    .bp3-tab-list > *:not(:last-child){
      margin-right:20px; }
  
  .bp3-tab{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    color:#182026;
    cursor:pointer;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    font-size:14px;
    line-height:30px;
    max-width:100%;
    position:relative;
    vertical-align:top; }
    .bp3-tab a{
      color:inherit;
      display:block;
      text-decoration:none; }
    .bp3-tab-indicator-wrapper ~ .bp3-tab{
      background-color:transparent !important;
      -webkit-box-shadow:none !important;
              box-shadow:none !important; }
    .bp3-tab[aria-disabled="true"]{
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
    .bp3-tab[aria-selected="true"]{
      border-radius:0;
      -webkit-box-shadow:inset 0 -3px 0 #106ba3;
              box-shadow:inset 0 -3px 0 #106ba3; }
    .bp3-tab[aria-selected="true"], .bp3-tab:not([aria-disabled="true"]):hover{
      color:#106ba3; }
    .bp3-tab:focus{
      -moz-outline-radius:0; }
    .bp3-large > .bp3-tab{
      font-size:16px;
      line-height:40px; }
  
  .bp3-tab-panel{
    margin-top:20px; }
    .bp3-tab-panel[aria-hidden="true"]{
      display:none; }
  
  .bp3-tab-indicator-wrapper{
    left:0;
    pointer-events:none;
    position:absolute;
    top:0;
    -webkit-transform:translateX(0), translateY(0);
            transform:translateX(0), translateY(0);
    -webkit-transition:height, width, -webkit-transform;
    transition:height, width, -webkit-transform;
    transition:height, transform, width;
    transition:height, transform, width, -webkit-transform;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-tab-indicator-wrapper .bp3-tab-indicator{
      background-color:#106ba3;
      bottom:0;
      height:3px;
      left:0;
      position:absolute;
      right:0; }
    .bp3-tab-indicator-wrapper.bp3-no-animation{
      -webkit-transition:none;
      transition:none; }
  
  .bp3-dark .bp3-tab{
    color:#f5f8fa; }
    .bp3-dark .bp3-tab[aria-disabled="true"]{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tab[aria-selected="true"]{
      -webkit-box-shadow:inset 0 -3px 0 #48aff0;
              box-shadow:inset 0 -3px 0 #48aff0; }
    .bp3-dark .bp3-tab[aria-selected="true"], .bp3-dark .bp3-tab:not([aria-disabled="true"]):hover{
      color:#48aff0; }
  
  .bp3-dark .bp3-tab-indicator{
    background-color:#48aff0; }
  
  .bp3-flex-expander{
    -webkit-box-flex:1;
        -ms-flex:1 1;
            flex:1 1; }
  .bp3-tag{
    display:-webkit-inline-box;
    display:-ms-inline-flexbox;
    display:inline-flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    background-color:#5c7080;
    border:none;
    border-radius:3px;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#f5f8fa;
    font-size:12px;
    line-height:16px;
    max-width:100%;
    min-height:20px;
    min-width:20px;
    padding:2px 6px;
    position:relative; }
    .bp3-tag.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-interactive:hover{
        background-color:rgba(92, 112, 128, 0.85); }
      .bp3-tag.bp3-interactive.bp3-active, .bp3-tag.bp3-interactive:active{
        background-color:rgba(92, 112, 128, 0.7); }
    .bp3-tag > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag::before,
    .bp3-tag > *{
      margin-right:4px; }
    .bp3-tag:empty::before,
    .bp3-tag > :last-child{
      margin-right:0; }
    .bp3-tag:focus{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:0;
      -moz-outline-radius:6px; }
    .bp3-tag.bp3-round{
      border-radius:30px;
      padding-left:8px;
      padding-right:8px; }
    .bp3-dark .bp3-tag{
      background-color:#bfccd6;
      color:#182026; }
      .bp3-dark .bp3-tag.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-interactive:hover{
          background-color:rgba(191, 204, 214, 0.85); }
        .bp3-dark .bp3-tag.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-interactive:active{
          background-color:rgba(191, 204, 214, 0.7); }
      .bp3-dark .bp3-tag > .bp3-icon, .bp3-dark .bp3-tag .bp3-icon-standard, .bp3-dark .bp3-tag .bp3-icon-large{
        fill:currentColor; }
    .bp3-tag > .bp3-icon, .bp3-tag .bp3-icon-standard, .bp3-tag .bp3-icon-large{
      fill:#ffffff; }
    .bp3-tag.bp3-large,
    .bp3-large .bp3-tag{
      font-size:14px;
      line-height:20px;
      min-height:30px;
      min-width:30px;
      padding:5px 10px; }
      .bp3-tag.bp3-large::before,
      .bp3-tag.bp3-large > *,
      .bp3-large .bp3-tag::before,
      .bp3-large .bp3-tag > *{
        margin-right:7px; }
      .bp3-tag.bp3-large:empty::before,
      .bp3-tag.bp3-large > :last-child,
      .bp3-large .bp3-tag:empty::before,
      .bp3-large .bp3-tag > :last-child{
        margin-right:0; }
      .bp3-tag.bp3-large.bp3-round,
      .bp3-large .bp3-tag.bp3-round{
        padding-left:12px;
        padding-right:12px; }
    .bp3-tag.bp3-intent-primary{
      background:#137cbd;
      color:#ffffff; }
      .bp3-tag.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.85); }
        .bp3-tag.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.7); }
    .bp3-tag.bp3-intent-success{
      background:#0f9960;
      color:#ffffff; }
      .bp3-tag.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.85); }
        .bp3-tag.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.7); }
    .bp3-tag.bp3-intent-warning{
      background:#d9822b;
      color:#ffffff; }
      .bp3-tag.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.85); }
        .bp3-tag.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.7); }
    .bp3-tag.bp3-intent-danger{
      background:#db3737;
      color:#ffffff; }
      .bp3-tag.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.85); }
        .bp3-tag.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.7); }
    .bp3-tag.bp3-fill{
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex;
      width:100%; }
    .bp3-tag.bp3-minimal > .bp3-icon, .bp3-tag.bp3-minimal .bp3-icon-standard, .bp3-tag.bp3-minimal .bp3-icon-large{
      fill:#5c7080; }
    .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
      background-color:rgba(138, 155, 168, 0.2);
      color:#182026; }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
          background-color:rgba(92, 112, 128, 0.3); }
        .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
          background-color:rgba(92, 112, 128, 0.4); }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
        color:#f5f8fa; }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
          cursor:pointer; }
          .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
            background-color:rgba(191, 204, 214, 0.3); }
          .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
            background-color:rgba(191, 204, 214, 0.4); }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) > .bp3-icon, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-large{
          fill:#a7b6c2; }
    .bp3-tag.bp3-minimal.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.15);
      color:#106ba3; }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.25); }
        .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.35); }
      .bp3-tag.bp3-minimal.bp3-intent-primary > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-large{
        fill:#137cbd; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary{
        background-color:rgba(19, 124, 189, 0.25);
        color:#48aff0; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
          cursor:pointer; }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
            background-color:rgba(19, 124, 189, 0.35); }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
            background-color:rgba(19, 124, 189, 0.45); }
    .bp3-tag.bp3-minimal.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.15);
      color:#0d8050; }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.25); }
        .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.35); }
      .bp3-tag.bp3-minimal.bp3-intent-success > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-large{
        fill:#0f9960; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success{
        background-color:rgba(15, 153, 96, 0.25);
        color:#3dcc91; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
          cursor:pointer; }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
            background-color:rgba(15, 153, 96, 0.35); }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
            background-color:rgba(15, 153, 96, 0.45); }
    .bp3-tag.bp3-minimal.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.15);
      color:#bf7326; }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.25); }
        .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.35); }
      .bp3-tag.bp3-minimal.bp3-intent-warning > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-large{
        fill:#d9822b; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning{
        background-color:rgba(217, 130, 43, 0.25);
        color:#ffb366; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
          cursor:pointer; }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
            background-color:rgba(217, 130, 43, 0.35); }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
            background-color:rgba(217, 130, 43, 0.45); }
    .bp3-tag.bp3-minimal.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.15);
      color:#c23030; }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.25); }
        .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.35); }
      .bp3-tag.bp3-minimal.bp3-intent-danger > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-large{
        fill:#db3737; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger{
        background-color:rgba(219, 55, 55, 0.25);
        color:#ff7373; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
          cursor:pointer; }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
            background-color:rgba(219, 55, 55, 0.35); }
          .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
            background-color:rgba(219, 55, 55, 0.45); }
  
  .bp3-tag-remove{
    background:none;
    border:none;
    color:inherit;
    cursor:pointer;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    margin-bottom:-2px;
    margin-right:-6px !important;
    margin-top:-2px;
    opacity:0.5;
    padding:2px;
    padding-left:0; }
    .bp3-tag-remove:hover{
      background:none;
      opacity:0.8;
      text-decoration:none; }
    .bp3-tag-remove:active{
      opacity:1; }
    .bp3-tag-remove:empty::before{
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      content:""; }
    .bp3-large .bp3-tag-remove{
      margin-right:-10px !important;
      padding:0 5px 0 0; }
      .bp3-large .bp3-tag-remove:empty::before{
        font-family:"Icons20", sans-serif;
        font-size:20px;
        font-style:normal;
        font-weight:400;
        line-height:1; }
  .bp3-tag-input{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    cursor:text;
    height:auto;
    line-height:inherit;
    min-height:30px;
    padding-left:5px;
    padding-right:0; }
    .bp3-tag-input > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag-input > .bp3-tag-input-values{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag-input .bp3-tag-input-icon{
      color:#5c7080;
      margin-left:2px;
      margin-right:7px;
      margin-top:7px; }
    .bp3-tag-input .bp3-tag-input-values{
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex;
      -webkit-box-orient:horizontal;
      -webkit-box-direction:normal;
          -ms-flex-direction:row;
              flex-direction:row;
      -webkit-box-align:center;
          -ms-flex-align:center;
              align-items:center;
      -ms-flex-item-align:stretch;
          align-self:stretch;
      -ms-flex-wrap:wrap;
          flex-wrap:wrap;
      margin-right:7px;
      margin-top:5px;
      min-width:0; }
      .bp3-tag-input .bp3-tag-input-values > *{
        -webkit-box-flex:0;
            -ms-flex-positive:0;
                flex-grow:0;
        -ms-flex-negative:0;
            flex-shrink:0; }
      .bp3-tag-input .bp3-tag-input-values > .bp3-fill{
        -webkit-box-flex:1;
            -ms-flex-positive:1;
                flex-grow:1;
        -ms-flex-negative:1;
            flex-shrink:1; }
      .bp3-tag-input .bp3-tag-input-values::before,
      .bp3-tag-input .bp3-tag-input-values > *{
        margin-right:5px; }
      .bp3-tag-input .bp3-tag-input-values:empty::before,
      .bp3-tag-input .bp3-tag-input-values > :last-child{
        margin-right:0; }
      .bp3-tag-input .bp3-tag-input-values:first-child .bp3-input-ghost:first-child{
        padding-left:5px; }
      .bp3-tag-input .bp3-tag-input-values > *{
        margin-bottom:5px; }
    .bp3-tag-input .bp3-tag{
      overflow-wrap:break-word; }
      .bp3-tag-input .bp3-tag.bp3-active{
        outline:rgba(19, 124, 189, 0.6) auto 2px;
        outline-offset:0;
        -moz-outline-radius:6px; }
    .bp3-tag-input .bp3-input-ghost{
      -webkit-box-flex:1;
          -ms-flex:1 1 auto;
              flex:1 1 auto;
      line-height:20px;
      width:80px; }
      .bp3-tag-input .bp3-input-ghost:disabled, .bp3-tag-input .bp3-input-ghost.bp3-disabled{
        cursor:not-allowed; }
    .bp3-tag-input .bp3-button,
    .bp3-tag-input .bp3-spinner{
      margin:3px;
      margin-left:0; }
    .bp3-tag-input .bp3-button{
      min-height:24px;
      min-width:24px;
      padding:0 7px; }
    .bp3-tag-input.bp3-large{
      height:auto;
      min-height:40px; }
      .bp3-tag-input.bp3-large::before,
      .bp3-tag-input.bp3-large > *{
        margin-right:10px; }
      .bp3-tag-input.bp3-large:empty::before,
      .bp3-tag-input.bp3-large > :last-child{
        margin-right:0; }
      .bp3-tag-input.bp3-large .bp3-tag-input-icon{
        margin-left:5px;
        margin-top:10px; }
      .bp3-tag-input.bp3-large .bp3-input-ghost{
        line-height:30px; }
      .bp3-tag-input.bp3-large .bp3-button{
        min-height:30px;
        min-width:30px;
        padding:5px 10px;
        margin:5px;
        margin-left:0; }
      .bp3-tag-input.bp3-large .bp3-spinner{
        margin:8px;
        margin-left:0; }
    .bp3-tag-input.bp3-active{
      background-color:#ffffff;
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-tag-input.bp3-active.bp3-intent-primary{
        -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-tag-input.bp3-active.bp3-intent-success{
        -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-tag-input.bp3-active.bp3-intent-warning{
        -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
      .bp3-tag-input.bp3-active.bp3-intent-danger{
        -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-tag-input .bp3-tag-input-icon, .bp3-tag-input.bp3-dark .bp3-tag-input-icon{
      color:#a7b6c2; }
    .bp3-dark .bp3-tag-input .bp3-input-ghost, .bp3-tag-input.bp3-dark .bp3-input-ghost{
      color:#f5f8fa; }
      .bp3-dark .bp3-tag-input .bp3-input-ghost::-webkit-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-webkit-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-tag-input .bp3-input-ghost::-moz-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-moz-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-tag-input .bp3-input-ghost:-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost:-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-tag-input .bp3-input-ghost::-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-ms-input-placeholder{
        color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-tag-input .bp3-input-ghost::placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::placeholder{
        color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input.bp3-active, .bp3-tag-input.bp3-dark.bp3-active{
      background-color:rgba(16, 22, 26, 0.3);
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-primary, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-primary{
        -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-success, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-success{
        -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-warning, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-warning{
        -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-danger, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-danger{
        -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  
  .bp3-input-ghost{
    background:none;
    border:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    padding:0; }
    .bp3-input-ghost::-webkit-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input-ghost::-moz-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input-ghost:-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input-ghost::-ms-input-placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input-ghost::placeholder{
      color:rgba(92, 112, 128, 0.6);
      opacity:1; }
    .bp3-input-ghost:focus{
      outline:none !important; }
  .bp3-toast{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    background-color:#ffffff;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    margin:20px 0 0;
    max-width:500px;
    min-width:300px;
    pointer-events:all;
    position:relative !important; }
    .bp3-toast.bp3-toast-enter, .bp3-toast.bp3-toast-appear{
      -webkit-transform:translateY(-40px);
              transform:translateY(-40px); }
    .bp3-toast.bp3-toast-enter-active, .bp3-toast.bp3-toast-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:300ms;
              transition-duration:300ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
              transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
    .bp3-toast.bp3-toast-enter ~ .bp3-toast, .bp3-toast.bp3-toast-appear ~ .bp3-toast{
      -webkit-transform:translateY(-40px);
              transform:translateY(-40px); }
    .bp3-toast.bp3-toast-enter-active ~ .bp3-toast, .bp3-toast.bp3-toast-appear-active ~ .bp3-toast{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:300ms;
              transition-duration:300ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
              transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
    .bp3-toast.bp3-toast-exit{
      opacity:1;
      -webkit-filter:blur(0);
              filter:blur(0); }
    .bp3-toast.bp3-toast-exit-active{
      opacity:0;
      -webkit-filter:blur(10px);
              filter:blur(10px);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:300ms;
              transition-duration:300ms;
      -webkit-transition-property:opacity, -webkit-filter;
      transition-property:opacity, -webkit-filter;
      transition-property:opacity, filter;
      transition-property:opacity, filter, -webkit-filter;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-toast.bp3-toast-exit ~ .bp3-toast{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-toast.bp3-toast-exit-active ~ .bp3-toast{
      -webkit-transform:translateY(-40px);
              transform:translateY(-40px);
      -webkit-transition-delay:50ms;
              transition-delay:50ms;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-toast .bp3-button-group{
      -webkit-box-flex:0;
          -ms-flex:0 0 auto;
              flex:0 0 auto;
      padding:5px;
      padding-left:0; }
    .bp3-toast > .bp3-icon{
      color:#5c7080;
      margin:12px;
      margin-right:0; }
    .bp3-toast.bp3-dark,
    .bp3-dark .bp3-toast{
      background-color:#394b59;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
      .bp3-toast.bp3-dark > .bp3-icon,
      .bp3-dark .bp3-toast > .bp3-icon{
        color:#a7b6c2; }
    .bp3-toast[class*="bp3-intent-"] a{
      color:rgba(255, 255, 255, 0.7); }
      .bp3-toast[class*="bp3-intent-"] a:hover{
        color:#ffffff; }
    .bp3-toast[class*="bp3-intent-"] > .bp3-icon{
      color:#ffffff; }
    .bp3-toast[class*="bp3-intent-"] .bp3-button, .bp3-toast[class*="bp3-intent-"] .bp3-button::before,
    .bp3-toast[class*="bp3-intent-"] .bp3-button .bp3-icon, .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
      color:rgba(255, 255, 255, 0.7) !important; }
    .bp3-toast[class*="bp3-intent-"] .bp3-button:focus{
      outline-color:rgba(255, 255, 255, 0.5); }
    .bp3-toast[class*="bp3-intent-"] .bp3-button:hover{
      background-color:rgba(255, 255, 255, 0.15) !important;
      color:#ffffff !important; }
    .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
      background-color:rgba(255, 255, 255, 0.3) !important;
      color:#ffffff !important; }
    .bp3-toast[class*="bp3-intent-"] .bp3-button::after{
      background:rgba(255, 255, 255, 0.3) !important; }
    .bp3-toast.bp3-intent-primary{
      background-color:#137cbd;
      color:#ffffff; }
    .bp3-toast.bp3-intent-success{
      background-color:#0f9960;
      color:#ffffff; }
    .bp3-toast.bp3-intent-warning{
      background-color:#d9822b;
      color:#ffffff; }
    .bp3-toast.bp3-intent-danger{
      background-color:#db3737;
      color:#ffffff; }
  
  .bp3-toast-message{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    padding:11px;
    word-break:break-word; }
  
  .bp3-toast-container{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box !important;
    display:-ms-flexbox !important;
    display:flex !important;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    left:0;
    overflow:hidden;
    padding:0 20px 20px;
    pointer-events:none;
    position:fixed;
    right:0;
    z-index:40; }
    .bp3-toast-container.bp3-toast-container-top{
      top:0; }
    .bp3-toast-container.bp3-toast-container-bottom{
      bottom:0;
      -webkit-box-orient:vertical;
      -webkit-box-direction:reverse;
          -ms-flex-direction:column-reverse;
              flex-direction:column-reverse;
      top:auto; }
    .bp3-toast-container.bp3-toast-container-left{
      -webkit-box-align:start;
          -ms-flex-align:start;
              align-items:flex-start; }
    .bp3-toast-container.bp3-toast-container-right{
      -webkit-box-align:end;
          -ms-flex-align:end;
              align-items:flex-end; }
  
  .bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active),
  .bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active) ~ .bp3-toast, .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active),
  .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active) ~ .bp3-toast,
  .bp3-toast-container-bottom .bp3-toast.bp3-toast-exit-active ~ .bp3-toast,
  .bp3-toast-container-bottom .bp3-toast.bp3-toast-leave-active ~ .bp3-toast{
    -webkit-transform:translateY(60px);
            transform:translateY(60px); }
  .bp3-tooltip{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    -webkit-transform:scale(1);
            transform:scale(1); }
    .bp3-tooltip .bp3-popover-arrow{
      height:22px;
      position:absolute;
      width:22px; }
      .bp3-tooltip .bp3-popover-arrow::before{
        height:14px;
        margin:4px;
        width:14px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip{
      margin-bottom:11px;
      margin-top:-11px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
        bottom:-8px; }
        .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow svg{
          -webkit-transform:rotate(-90deg);
                  transform:rotate(-90deg); }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip{
      margin-left:11px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
        left:-8px; }
        .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow svg{
          -webkit-transform:rotate(0);
                  transform:rotate(0); }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip{
      margin-top:11px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
        top:-8px; }
        .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow svg{
          -webkit-transform:rotate(90deg);
                  transform:rotate(90deg); }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip{
      margin-left:-11px;
      margin-right:11px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
        right:-8px; }
        .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow svg{
          -webkit-transform:rotate(180deg);
                  transform:rotate(180deg); }
    .bp3-tether-element-attached-middle > .bp3-tooltip > .bp3-popover-arrow{
      top:50%;
      -webkit-transform:translateY(-50%);
              transform:translateY(-50%); }
    .bp3-tether-element-attached-center > .bp3-tooltip > .bp3-popover-arrow{
      right:50%;
      -webkit-transform:translateX(50%);
              transform:translateX(50%); }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
      top:-0.22183px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
      right:-0.22183px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
      left:-0.22183px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
      bottom:-0.22183px; }
    .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-tooltip{
      -webkit-transform-origin:top left;
              transform-origin:top left; }
    .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-tooltip{
      -webkit-transform-origin:top center;
              transform-origin:top center; }
    .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-tooltip{
      -webkit-transform-origin:top right;
              transform-origin:top right; }
    .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-tooltip{
      -webkit-transform-origin:center left;
              transform-origin:center left; }
    .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-tooltip{
      -webkit-transform-origin:center center;
              transform-origin:center center; }
    .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-tooltip{
      -webkit-transform-origin:center right;
              transform-origin:center right; }
    .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-tooltip{
      -webkit-transform-origin:bottom left;
              transform-origin:bottom left; }
    .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-tooltip{
      -webkit-transform-origin:bottom center;
              transform-origin:bottom center; }
    .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-tooltip{
      -webkit-transform-origin:bottom right;
              transform-origin:bottom right; }
    .bp3-tooltip .bp3-popover-content{
      background:#394b59;
      color:#f5f8fa; }
    .bp3-tooltip .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
    .bp3-tooltip .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.1; }
    .bp3-tooltip .bp3-popover-arrow-fill{
      fill:#394b59; }
    .bp3-popover-enter > .bp3-tooltip, .bp3-popover-appear > .bp3-tooltip{
      -webkit-transform:scale(0.8);
              transform:scale(0.8); }
    .bp3-popover-enter-active > .bp3-tooltip, .bp3-popover-appear-active > .bp3-tooltip{
      -webkit-transform:scale(1);
              transform:scale(1);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-popover-exit > .bp3-tooltip{
      -webkit-transform:scale(1);
              transform:scale(1); }
    .bp3-popover-exit-active > .bp3-tooltip{
      -webkit-transform:scale(0.8);
              transform:scale(0.8);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-tooltip .bp3-popover-content{
      padding:10px 12px; }
    .bp3-tooltip.bp3-dark,
    .bp3-dark .bp3-tooltip{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
      .bp3-tooltip.bp3-dark .bp3-popover-content,
      .bp3-dark .bp3-tooltip .bp3-popover-content{
        background:#e1e8ed;
        color:#394b59; }
      .bp3-tooltip.bp3-dark .bp3-popover-arrow::before,
      .bp3-dark .bp3-tooltip .bp3-popover-arrow::before{
        -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
                box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
      .bp3-tooltip.bp3-dark .bp3-popover-arrow-border,
      .bp3-dark .bp3-tooltip .bp3-popover-arrow-border{
        fill:#10161a;
        fill-opacity:0.2; }
      .bp3-tooltip.bp3-dark .bp3-popover-arrow-fill,
      .bp3-dark .bp3-tooltip .bp3-popover-arrow-fill{
        fill:#e1e8ed; }
    .bp3-tooltip.bp3-intent-primary .bp3-popover-content{
      background:#137cbd;
      color:#ffffff; }
    .bp3-tooltip.bp3-intent-primary .bp3-popover-arrow-fill{
      fill:#137cbd; }
    .bp3-tooltip.bp3-intent-success .bp3-popover-content{
      background:#0f9960;
      color:#ffffff; }
    .bp3-tooltip.bp3-intent-success .bp3-popover-arrow-fill{
      fill:#0f9960; }
    .bp3-tooltip.bp3-intent-warning .bp3-popover-content{
      background:#d9822b;
      color:#ffffff; }
    .bp3-tooltip.bp3-intent-warning .bp3-popover-arrow-fill{
      fill:#d9822b; }
    .bp3-tooltip.bp3-intent-danger .bp3-popover-content{
      background:#db3737;
      color:#ffffff; }
    .bp3-tooltip.bp3-intent-danger .bp3-popover-arrow-fill{
      fill:#db3737; }
  
  .bp3-tooltip-indicator{
    border-bottom:dotted 1px;
    cursor:help; }
  .bp3-tree .bp3-icon, .bp3-tree .bp3-icon-standard, .bp3-tree .bp3-icon-large{
    color:#5c7080; }
    .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-tree .bp3-icon-large.bp3-intent-primary{
      color:#137cbd; }
    .bp3-tree .bp3-icon.bp3-intent-success, .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-tree .bp3-icon-large.bp3-intent-success{
      color:#0f9960; }
    .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-tree .bp3-icon-large.bp3-intent-warning{
      color:#d9822b; }
    .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-tree .bp3-icon-large.bp3-intent-danger{
      color:#db3737; }
  
  .bp3-tree-node-list{
    list-style:none;
    margin:0;
    padding-left:0; }
  
  .bp3-tree-root{
    background-color:transparent;
    cursor:default;
    padding-left:0;
    position:relative; }
  
  .bp3-tree-node-content-0{
    padding-left:0px; }
  
  .bp3-tree-node-content-1{
    padding-left:23px; }
  
  .bp3-tree-node-content-2{
    padding-left:46px; }
  
  .bp3-tree-node-content-3{
    padding-left:69px; }
  
  .bp3-tree-node-content-4{
    padding-left:92px; }
  
  .bp3-tree-node-content-5{
    padding-left:115px; }
  
  .bp3-tree-node-content-6{
    padding-left:138px; }
  
  .bp3-tree-node-content-7{
    padding-left:161px; }
  
  .bp3-tree-node-content-8{
    padding-left:184px; }
  
  .bp3-tree-node-content-9{
    padding-left:207px; }
  
  .bp3-tree-node-content-10{
    padding-left:230px; }
  
  .bp3-tree-node-content-11{
    padding-left:253px; }
  
  .bp3-tree-node-content-12{
    padding-left:276px; }
  
  .bp3-tree-node-content-13{
    padding-left:299px; }
  
  .bp3-tree-node-content-14{
    padding-left:322px; }
  
  .bp3-tree-node-content-15{
    padding-left:345px; }
  
  .bp3-tree-node-content-16{
    padding-left:368px; }
  
  .bp3-tree-node-content-17{
    padding-left:391px; }
  
  .bp3-tree-node-content-18{
    padding-left:414px; }
  
  .bp3-tree-node-content-19{
    padding-left:437px; }
  
  .bp3-tree-node-content-20{
    padding-left:460px; }
  
  .bp3-tree-node-content{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    height:30px;
    padding-right:5px;
    width:100%; }
    .bp3-tree-node-content:hover{
      background-color:rgba(191, 204, 214, 0.4); }
  
  .bp3-tree-node-caret,
  .bp3-tree-node-caret-none{
    min-width:30px; }
  
  .bp3-tree-node-caret{
    color:#5c7080;
    cursor:pointer;
    padding:7px;
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg);
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-tree-node-caret:hover{
      color:#182026; }
    .bp3-dark .bp3-tree-node-caret{
      color:#a7b6c2; }
      .bp3-dark .bp3-tree-node-caret:hover{
        color:#f5f8fa; }
    .bp3-tree-node-caret.bp3-tree-node-caret-open{
      -webkit-transform:rotate(90deg);
              transform:rotate(90deg); }
    .bp3-tree-node-caret.bp3-icon-standard::before{
      content:""; }
  
  .bp3-tree-node-icon{
    margin-right:7px;
    position:relative; }
  
  .bp3-tree-node-label{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    position:relative;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-tree-node-label span{
      display:inline; }
  
  .bp3-tree-node-secondary-label{
    padding:0 5px;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none; }
    .bp3-tree-node-secondary-label .bp3-popover-wrapper,
    .bp3-tree-node-secondary-label .bp3-popover-target{
      -webkit-box-align:center;
          -ms-flex-align:center;
              align-items:center;
      display:-webkit-box;
      display:-ms-flexbox;
      display:flex; }
  
  .bp3-tree-node.bp3-disabled .bp3-tree-node-content{
    background-color:inherit;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  
  .bp3-tree-node.bp3-disabled .bp3-tree-node-caret,
  .bp3-tree-node.bp3-disabled .bp3-tree-node-icon{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
    background-color:#137cbd; }
    .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content,
    .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-standard, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-large{
      color:#ffffff; }
    .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret::before{
      color:rgba(255, 255, 255, 0.7); }
    .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret:hover::before{
      color:#ffffff; }
  
  .bp3-dark .bp3-tree-node-content:hover{
    background-color:rgba(92, 112, 128, 0.3); }
  
  .bp3-dark .bp3-tree .bp3-icon, .bp3-dark .bp3-tree .bp3-icon-standard, .bp3-dark .bp3-tree .bp3-icon-large{
    color:#a7b6c2; }
    .bp3-dark .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-primary{
      color:#137cbd; }
    .bp3-dark .bp3-tree .bp3-icon.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-success{
      color:#0f9960; }
    .bp3-dark .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-warning{
      color:#d9822b; }
    .bp3-dark .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-danger{
      color:#db3737; }
  
  .bp3-dark .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
    background-color:#137cbd; }
  .bp3-omnibar{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1;
    background-color:#ffffff;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
    left:calc(50% - 250px);
    top:20vh;
    width:500px;
    z-index:21; }
    .bp3-omnibar.bp3-overlay-enter, .bp3-omnibar.bp3-overlay-appear{
      -webkit-filter:blur(20px);
              filter:blur(20px);
      opacity:0.2; }
    .bp3-omnibar.bp3-overlay-enter-active, .bp3-omnibar.bp3-overlay-appear-active{
      -webkit-filter:blur(0);
              filter:blur(0);
      opacity:1;
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:opacity, -webkit-filter;
      transition-property:opacity, -webkit-filter;
      transition-property:filter, opacity;
      transition-property:filter, opacity, -webkit-filter;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-omnibar.bp3-overlay-exit{
      -webkit-filter:blur(0);
              filter:blur(0);
      opacity:1; }
    .bp3-omnibar.bp3-overlay-exit-active{
      -webkit-filter:blur(20px);
              filter:blur(20px);
      opacity:0.2;
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:opacity, -webkit-filter;
      transition-property:opacity, -webkit-filter;
      transition-property:filter, opacity;
      transition-property:filter, opacity, -webkit-filter;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-omnibar .bp3-input{
      background-color:transparent;
      border-radius:0; }
      .bp3-omnibar .bp3-input, .bp3-omnibar .bp3-input:focus{
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-omnibar .bp3-menu{
      background-color:transparent;
      border-radius:0;
      -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
              box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
      max-height:calc(60vh - 40px);
      overflow:auto; }
      .bp3-omnibar .bp3-menu:empty{
        display:none; }
    .bp3-dark .bp3-omnibar, .bp3-omnibar.bp3-dark{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }
  
  .bp3-omnibar-overlay .bp3-overlay-backdrop{
    background-color:rgba(16, 22, 26, 0.2); }
  
  .bp3-select-popover .bp3-popover-content{
    padding:5px; }
  
  .bp3-select-popover .bp3-input-group{
    margin-bottom:0; }
  
  .bp3-select-popover .bp3-menu{
    max-height:300px;
    max-width:400px;
    overflow:auto;
    padding:0; }
    .bp3-select-popover .bp3-menu:not(:first-child){
      padding-top:5px; }
  
  .bp3-multi-select{
    min-width:150px; }
  
  .bp3-multi-select-popover .bp3-menu{
    max-height:300px;
    max-width:400px;
    overflow:auto; }
  
  .bp3-select-popover .bp3-popover-content{
    padding:5px; }
  
  .bp3-select-popover .bp3-input-group{
    margin-bottom:0; }
  
  .bp3-select-popover .bp3-menu{
    max-height:300px;
    max-width:400px;
    overflow:auto;
    padding:0; }
    .bp3-select-popover .bp3-menu:not(:first-child){
      padding-top:5px; }
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */
  
  /**
   * (DEPRECATED) Support for consuming icons as CSS background images
   */
  
  /* Icons urls */
  
  :root {
    --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDhoLTIuODFjLS40NS0uNzgtMS4wNy0xLjQ1LTEuODItMS45NkwxNyA0LjQxIDE1LjU5IDNsLTIuMTcgMi4xN0MxMi45NiA1LjA2IDEyLjQ5IDUgMTIgNWMtLjQ5IDAtLjk2LjA2LTEuNDEuMTdMOC40MSAzIDcgNC40MWwxLjYyIDEuNjNDNy44OCA2LjU1IDcuMjYgNy4yMiA2LjgxIDhINHYyaDIuMDljLS4wNS4zMy0uMDkuNjYtLjA5IDF2MUg0djJoMnYxYzAgLjM0LjA0LjY3LjA5IDFINHYyaDIuODFjMS4wNCAxLjc5IDIuOTcgMyA1LjE5IDNzNC4xNS0xLjIxIDUuMTktM0gyMHYtMmgtMi4wOWMuMDUtLjMzLjA5LS42Ni4wOS0xdi0xaDJ2LTJoLTJ2LTFjMC0uMzQtLjA0LS42Ny0uMDktMUgyMFY4em0tNiA4aC00di0yaDR2MnptMC00aC00di0yaDR2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
    --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
    --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTYuMTdMNC44MyAxMmwtMS40MiAxLjQxTDkgMTkgMjEgN2wtMS40MS0xLjQxeiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
    --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
    --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
    --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
    --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
    --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
    --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNGOUE4MjUiPgogICAgPHBhdGggZD0iTTIwLjIgMTEuOGMtMS42IDAtMS43LjUtMS43IDEgMCAuNC4xLjkuMSAxLjMuMS41LjEuOS4xIDEuMyAwIDEuNy0xLjQgMi4zLTMuNSAyLjNoLS45di0xLjloLjVjMS4xIDAgMS40IDAgMS40LS44IDAtLjMgMC0uNi0uMS0xIDAtLjQtLjEtLjgtLjEtMS4yIDAtMS4zIDAtMS44IDEuMy0yLTEuMy0uMi0xLjMtLjctMS4zLTIgMC0uNC4xLS44LjEtMS4yLjEtLjQuMS0uNy4xLTEgMC0uOC0uNC0uNy0xLjQtLjhoLS41VjQuMWguOWMyLjIgMCAzLjUuNyAzLjUgMi4zIDAgLjQtLjEuOS0uMSAxLjMtLjEuNS0uMS45LS4xIDEuMyAwIC41LjIgMSAxLjcgMXYxLjh6TTEuOCAxMC4xYzEuNiAwIDEuNy0uNSAxLjctMSAwLS40LS4xLS45LS4xLTEuMy0uMS0uNS0uMS0uOS0uMS0xLjMgMC0xLjYgMS40LTIuMyAzLjUtMi4zaC45djEuOWgtLjVjLTEgMC0xLjQgMC0xLjQuOCAwIC4zIDAgLjYuMSAxIDAgLjIuMS42LjEgMSAwIDEuMyAwIDEuOC0xLjMgMkM2IDExLjIgNiAxMS43IDYgMTNjMCAuNC0uMS44LS4xIDEuMi0uMS4zLS4xLjctLjEgMSAwIC44LjMuOCAxLjQuOGguNXYxLjloLS45Yy0yLjEgMC0zLjUtLjYtMy41LTIuMyAwLS40LjEtLjkuMS0xLjMuMS0uNS4xLS45LjEtMS4zIDAtLjUtLjItMS0xLjctMXYtMS45eiIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSIxMy44IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY3g9IjExIiBjeT0iOC4yIiByPSIyLjEiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
    --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
    --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
    --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
    --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
    --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
    --jp-icon-listings-info: url(data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJDYXBhXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4Ig0KCSB2aWV3Qm94PSIwIDAgNTAuOTc4IDUwLjk3OCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgNTAuOTc4IDUwLjk3ODsiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxnPg0KCQkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik00My41Miw3LjQ1OEMzOC43MTEsMi42NDgsMzIuMzA3LDAsMjUuNDg5LDBDMTguNjcsMCwxMi4yNjYsMi42NDgsNy40NTgsNy40NTgNCgkJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDANCgkJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoNCgkJCQkgTTQyLjEwNiw0Mi4xMDVjLTQuNDMyLDQuNDMxLTEwLjMzMiw2Ljg3Mi0xNi42MTUsNi44NzJoLTAuMDAyYy02LjI4NS0wLjAwMS0xMi4xODctMi40NDEtMTYuNjE3LTYuODcyDQoJCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzINCgkJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4NCgkJPC9nPg0KCQk8Zz4NCgkJCTxwYXRoIHN0eWxlPSJmaWxsOiMwMTAwMDI7IiBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1Mw0KCQkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUNCgkJCQljMC0xLjA5Ni0wLjI2LTIuMDg4LTAuNzc5LTIuOTc5Yy0wLjU2NS0wLjg3OS0xLjUwMS0xLjMzNi0yLjgwNi0xLjM2OWMtMS44MDIsMC4wNTctMi45ODUsMC42NjctMy41NSwxLjgzMg0KCQkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkNCgkJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQ0KCQkJCWMwLDEuMTQyLTAuMTM3LDIuMTExLTAuNDEsMi45MTFjLTAuMzA5LDAuODQ1LTAuNzMxLDEuNTkzLTEuMjY4LDIuMjQzYy0wLjQ5MiwwLjY1LTEuMDY4LDEuMzE4LTEuNzMsMi4wMDINCgkJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5DQoJCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+DQoJCTwvZz4NCgk8L2c+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8L3N2Zz4NCg==);
    --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
    --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
    --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
    --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
    --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMEQ0N0ExIj4KICAgIDxwYXRoIGQ9Ik0xMS4xIDYuOVY1LjhINi45YzAtLjUgMC0xLjMuMi0xLjYuNC0uNy44LTEuMSAxLjctMS40IDEuNy0uMyAyLjUtLjMgMy45LS4xIDEgLjEgMS45LjkgMS45IDEuOXY0LjJjMCAuNS0uOSAxLjYtMiAxLjZIOC44Yy0xLjUgMC0yLjQgMS40LTIuNCAyLjh2Mi4ySDQuN0MzLjUgMTUuMSAzIDE0IDMgMTMuMVY5Yy0uMS0xIC42LTIgMS44LTIgMS41LS4xIDYuMy0uMSA2LjMtLjF6Ii8+CiAgICA8cGF0aCBkPSJNMTAuOSAxNS4xdjEuMWg0LjJjMCAuNSAwIDEuMy0uMiAxLjYtLjQuNy0uOCAxLjEtMS43IDEuNC0xLjcuMy0yLjUuMy0zLjkuMS0xLS4xLTEuOS0uOS0xLjktMS45di00LjJjMC0uNS45LTEuNiAyLTEuNmgzLjhjMS41IDAgMi40LTEuNCAyLjQtMi44VjYuNmgxLjdDMTguNSA2LjkgMTkgOCAxOSA4LjlWMTNjMCAxLS43IDIuMS0xLjkgMi4xaC02LjJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
    --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
    --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
    --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
    --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
    --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
    --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
    --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4=);
    --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
    --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIGZpbGw9IiMzMzMzMzMiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uLWFjY2VudDIganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGQ9Ik01LjA1NjY0IDguNzYxNzJDNS4wNTY2NCA4LjU5NzY2IDUuMDMxMjUgOC40NTMxMiA0Ljk4MDQ3IDguMzI4MTJDNC45MzM1OSA4LjE5OTIyIDQuODU1NDcgOC4wODIwMyA0Ljc0NjA5IDcuOTc2NTZDNC42NDA2MiA3Ljg3MTA5IDQuNSA3Ljc3NTM5IDQuMzI0MjIgNy42ODk0NUM0LjE1MjM0IDcuNTk5NjEgMy45NDMzNiA3LjUxMTcyIDMuNjk3MjcgNy40MjU3OEMzLjMwMjczIDcuMjg1MTYgMi45NDMzNiA3LjEzNjcyIDIuNjE5MTQgNi45ODA0N0MyLjI5NDkyIDYuODI0MjIgMi4wMTc1OCA2LjY0MjU4IDEuNzg3MTEgNi40MzU1NUMxLjU2MDU1IDYuMjI4NTIgMS4zODQ3NyA1Ljk4ODI4IDEuMjU5NzcgNS43MTQ4NEMxLjEzNDc3IDUuNDM3NSAxLjA3MjI3IDUuMTA5MzggMS4wNzIyNyA0LjczMDQ3QzEuMDcyMjcgNC4zOTg0NCAxLjEyODkxIDQuMDk1NyAxLjI0MjE5IDMuODIyMjdDMS4zNTU0NyAzLjU0NDkyIDEuNTE1NjIgMy4zMDQ2OSAxLjcyMjY2IDMuMTAxNTZDMS45Mjk2OSAyLjg5ODQ0IDIuMTc5NjkgMi43MzQzNyAyLjQ3MjY2IDIuNjA5MzhDMi43NjU2MiAyLjQ4NDM4IDMuMDkxOCAyLjQwNDMgMy40NTExNyAyLjM2OTE0VjEuMTA5MzhINC4zODg2N1YyLjM4MDg2QzQuNzQwMjMgMi40Mjc3MyA1LjA1NjY0IDIuNTIzNDQgNS4zMzc4OSAyLjY2Nzk3QzUuNjE5MTQgMi44MTI1IDUuODU3NDIgMy4wMDE5NSA2LjA1MjczIDMuMjM2MzNDNi4yNTE5NSAzLjQ2NjggNi40MDQzIDMuNzQwMjMgNi41MDk3NyA0LjA1NjY0QzYuNjE5MTQgNC4zNjkxNCA2LjY3MzgzIDQuNzIwNyA2LjY3MzgzIDUuMTExMzNINS4wNDQ5MkM1LjA0NDkyIDQuNjM4NjcgNC45Mzc1IDQuMjgxMjUgNC43MjI2NiA0LjAzOTA2QzQuNTA3ODEgMy43OTI5NyA0LjIxNjggMy42Njk5MiAzLjg0OTYxIDMuNjY5OTJDMy42NTAzOSAzLjY2OTkyIDMuNDc2NTYgMy42OTcyNyAzLjMyODEyIDMuNzUxOTVDMy4xODM1OSAzLjgwMjczIDMuMDY0NDUgMy44NzY5NSAyLjk3MDcgMy45NzQ2MUMyLjg3Njk1IDQuMDY4MzYgMi44MDY2NCA0LjE3OTY5IDIuNzU5NzcgNC4zMDg1OUMyLjcxNjggNC40Mzc1IDIuNjk1MzEgNC41NzgxMiAyLjY5NTMxIDQuNzMwNDdDMi42OTUzMSA0Ljg4MjgxIDIuNzE2OCA1LjAxOTUzIDIuNzU5NzcgNS4xNDA2MkMyLjgwNjY0IDUuMjU3ODEgMi44ODI4MSA1LjM2NzE5IDIuOTg4MjggNS40Njg3NUMzLjA5NzY2IDUuNTcwMzEgMy4yNDAyMyA1LjY2Nzk3IDMuNDE2MDIgNS43NjE3MkMzLjU5MTggNS44NTE1NiAzLjgxMDU1IDUuOTQzMzYgNC4wNzIyNyA2LjAzNzExQzQuNDY2OCA2LjE4NTU1IDQuODI0MjIgNi4zMzk4NCA1LjE0NDUzIDYuNUM1LjQ2NDg0IDYuNjU2MjUgNS43MzgyOCA2LjgzOTg0IDUuOTY0ODQgNy4wNTA3OEM2LjE5NTMxIDcuMjU3ODEgNi4zNzEwOSA3LjUgNi40OTIxOSA3Ljc3NzM0QzYuNjE3MTkgOC4wNTA3OCA2LjY3OTY5IDguMzc1IDYuNjc5NjkgOC43NUM2LjY3OTY5IDkuMDkzNzUgNi42MjMwNSA5LjQwNDMgNi41MDk3NyA5LjY4MTY0QzYuMzk2NDggOS45NTUwOCA2LjIzNDM4IDEwLjE5MTQgNi4wMjM0NCAxMC4zOTA2QzUuODEyNSAxMC41ODk4IDUuNTU4NTkgMTAuNzUgNS4yNjE3MiAxMC44NzExQzQuOTY0ODQgMTAuOTg4MyA0LjYzMjgxIDExLjA2NDUgNC4yNjU2MiAxMS4wOTk2VjEyLjI0OEgzLjMzMzk4VjExLjA5OTZDMy4wMDE5NSAxMS4wNjg0IDIuNjc5NjkgMTAuOTk2MSAyLjM2NzE5IDEwLjg4MjhDMi4wNTQ2OSAxMC43NjU2IDEuNzc3MzQgMTAuNTk3NyAxLjUzNTE2IDEwLjM3ODlDMS4yOTY4OCAxMC4xNjAyIDEuMTA1NDcgOS44ODQ3NyAwLjk2MDkzOCA5LjU1MjczQzAuODE2NDA2IDkuMjE2OCAwLjc0NDE0MSA4LjgxNDQ1IDAuNzQ0MTQxIDguMzQ1N0gyLjM3ODkxQzIuMzc4OTEgOC42MjY5NSAyLjQxOTkyIDguODYzMjggMi41MDE5NSA5LjA1NDY5QzIuNTgzOTggOS4yNDIxOSAyLjY4OTQ1IDkuMzkyNTggMi44MTgzNiA5LjUwNTg2QzIuOTUxMTcgOS42MTUyMyAzLjEwMTU2IDkuNjkzMzYgMy4yNjk1MyA5Ljc0MDIzQzMuNDM3NSA5Ljc4NzExIDMuNjA5MzggOS44MTA1NSAzLjc4NTE2IDkuODEwNTVDNC4yMDMxMiA5LjgxMDU1IDQuNTE5NTMgOS43MTI4OSA0LjczNDM4IDkuNTE3NThDNC45NDkyMiA5LjMyMjI3IDUuMDU2NjQgOS4wNzAzMSA1LjA1NjY0IDguNzYxNzJaTTEzLjQxOCAxMi4yNzE1SDguMDc0MjJWMTFIMTMuNDE4VjEyLjI3MTVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzLjk1MjY0IDYpIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
    --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTUgMTVIM3YyaDEydi0yem0wLThIM3YyaDEyVjd6TTMgMTNoMTh2LTJIM3Yyem0wIDhoMTh2LTJIM3Yyek0zIDN2MmgxOFYzSDN6Ii8+Cjwvc3ZnPgo=);
    --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2ZXJzaW9uPSIxLjEiIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgoJPHBhdGggZD0iTTcsNUgyMVY3SDdWNU03LDEzVjExSDIxVjEzSDdNNCw0LjVBMS41LDEuNSAwIDAsMSA1LjUsNkExLjUsMS41IDAgMCwxIDQsNy41QTEuNSwxLjUgMCAwLDEgMi41LDZBMS41LDEuNSAwIDAsMSA0LDQuNU00LDEwLjVBMS41LDEuNSAwIDAsMSA1LjUsMTJBMS41LDEuNSAwIDAsMSA0LDEzLjVBMS41LDEuNSAwIDAsMSAyLjUsMTJBMS41LDEuNSAwIDAsMSA0LDEwLjVNNywxOVYxN0gyMVYxOUg3TTQsMTYuNUExLjUsMS41IDAgMCwxIDUuNSwxOEExLjUsMS41IDAgMCwxIDQsMTkuNUExLjUsMS41IDAgMCwxIDIuNSwxOEExLjUsMS41IDAgMCwxIDQsMTYuNVoiIC8+Cjwvc3ZnPgo=);
    --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4=);
    --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
    --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
    --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
  }
  
  /* Icon CSS class declarations */
  
  .jp-AddIcon {
    background-image: var(--jp-icon-add);
  }
  .jp-BugIcon {
    background-image: var(--jp-icon-bug);
  }
  .jp-BuildIcon {
    background-image: var(--jp-icon-build);
  }
  .jp-CaretDownEmptyIcon {
    background-image: var(--jp-icon-caret-down-empty);
  }
  .jp-CaretDownEmptyThinIcon {
    background-image: var(--jp-icon-caret-down-empty-thin);
  }
  .jp-CaretDownIcon {
    background-image: var(--jp-icon-caret-down);
  }
  .jp-CaretLeftIcon {
    background-image: var(--jp-icon-caret-left);
  }
  .jp-CaretRightIcon {
    background-image: var(--jp-icon-caret-right);
  }
  .jp-CaretUpEmptyThinIcon {
    background-image: var(--jp-icon-caret-up-empty-thin);
  }
  .jp-CaretUpIcon {
    background-image: var(--jp-icon-caret-up);
  }
  .jp-CaseSensitiveIcon {
    background-image: var(--jp-icon-case-sensitive);
  }
  .jp-CheckIcon {
    background-image: var(--jp-icon-check);
  }
  .jp-CircleEmptyIcon {
    background-image: var(--jp-icon-circle-empty);
  }
  .jp-CircleIcon {
    background-image: var(--jp-icon-circle);
  }
  .jp-ClearIcon {
    background-image: var(--jp-icon-clear);
  }
  .jp-CloseIcon {
    background-image: var(--jp-icon-close);
  }
  .jp-CodeIcon {
    background-image: var(--jp-icon-code);
  }
  .jp-ConsoleIcon {
    background-image: var(--jp-icon-console);
  }
  .jp-CopyIcon {
    background-image: var(--jp-icon-copy);
  }
  .jp-CutIcon {
    background-image: var(--jp-icon-cut);
  }
  .jp-DownloadIcon {
    background-image: var(--jp-icon-download);
  }
  .jp-EditIcon {
    background-image: var(--jp-icon-edit);
  }
  .jp-EllipsesIcon {
    background-image: var(--jp-icon-ellipses);
  }
  .jp-ExtensionIcon {
    background-image: var(--jp-icon-extension);
  }
  .jp-FastForwardIcon {
    background-image: var(--jp-icon-fast-forward);
  }
  .jp-FileIcon {
    background-image: var(--jp-icon-file);
  }
  .jp-FileUploadIcon {
    background-image: var(--jp-icon-file-upload);
  }
  .jp-FilterListIcon {
    background-image: var(--jp-icon-filter-list);
  }
  .jp-FolderIcon {
    background-image: var(--jp-icon-folder);
  }
  .jp-Html5Icon {
    background-image: var(--jp-icon-html5);
  }
  .jp-ImageIcon {
    background-image: var(--jp-icon-image);
  }
  .jp-InspectorIcon {
    background-image: var(--jp-icon-inspector);
  }
  .jp-JsonIcon {
    background-image: var(--jp-icon-json);
  }
  .jp-JupyterFaviconIcon {
    background-image: var(--jp-icon-jupyter-favicon);
  }
  .jp-JupyterIcon {
    background-image: var(--jp-icon-jupyter);
  }
  .jp-JupyterlabWordmarkIcon {
    background-image: var(--jp-icon-jupyterlab-wordmark);
  }
  .jp-KernelIcon {
    background-image: var(--jp-icon-kernel);
  }
  .jp-KeyboardIcon {
    background-image: var(--jp-icon-keyboard);
  }
  .jp-LauncherIcon {
    background-image: var(--jp-icon-launcher);
  }
  .jp-LineFormIcon {
    background-image: var(--jp-icon-line-form);
  }
  .jp-LinkIcon {
    background-image: var(--jp-icon-link);
  }
  .jp-ListIcon {
    background-image: var(--jp-icon-list);
  }
  .jp-ListingsInfoIcon {
    background-image: var(--jp-icon-listings-info);
  }
  .jp-MarkdownIcon {
    background-image: var(--jp-icon-markdown);
  }
  .jp-NewFolderIcon {
    background-image: var(--jp-icon-new-folder);
  }
  .jp-NotTrustedIcon {
    background-image: var(--jp-icon-not-trusted);
  }
  .jp-NotebookIcon {
    background-image: var(--jp-icon-notebook);
  }
  .jp-NumberingIcon {
    background-image: var(--jp-icon-numbering);
  }
  .jp-OfflineBoltIcon {
    background-image: var(--jp-icon-offline-bolt);
  }
  .jp-PaletteIcon {
    background-image: var(--jp-icon-palette);
  }
  .jp-PasteIcon {
    background-image: var(--jp-icon-paste);
  }
  .jp-PdfIcon {
    background-image: var(--jp-icon-pdf);
  }
  .jp-PythonIcon {
    background-image: var(--jp-icon-python);
  }
  .jp-RKernelIcon {
    background-image: var(--jp-icon-r-kernel);
  }
  .jp-ReactIcon {
    background-image: var(--jp-icon-react);
  }
  .jp-RedoIcon {
    background-image: var(--jp-icon-redo);
  }
  .jp-RefreshIcon {
    background-image: var(--jp-icon-refresh);
  }
  .jp-RegexIcon {
    background-image: var(--jp-icon-regex);
  }
  .jp-RunIcon {
    background-image: var(--jp-icon-run);
  }
  .jp-RunningIcon {
    background-image: var(--jp-icon-running);
  }
  .jp-SaveIcon {
    background-image: var(--jp-icon-save);
  }
  .jp-SearchIcon {
    background-image: var(--jp-icon-search);
  }
  .jp-SettingsIcon {
    background-image: var(--jp-icon-settings);
  }
  .jp-SpreadsheetIcon {
    background-image: var(--jp-icon-spreadsheet);
  }
  .jp-StopIcon {
    background-image: var(--jp-icon-stop);
  }
  .jp-TabIcon {
    background-image: var(--jp-icon-tab);
  }
  .jp-TableRowsIcon {
    background-image: var(--jp-icon-table-rows);
  }
  .jp-TagIcon {
    background-image: var(--jp-icon-tag);
  }
  .jp-TerminalIcon {
    background-image: var(--jp-icon-terminal);
  }
  .jp-TextEditorIcon {
    background-image: var(--jp-icon-text-editor);
  }
  .jp-TocIcon {
    background-image: var(--jp-icon-toc);
  }
  .jp-TreeViewIcon {
    background-image: var(--jp-icon-tree-view);
  }
  .jp-TrustedIcon {
    background-image: var(--jp-icon-trusted);
  }
  .jp-UndoIcon {
    background-image: var(--jp-icon-undo);
  }
  .jp-VegaIcon {
    background-image: var(--jp-icon-vega);
  }
  .jp-YamlIcon {
    background-image: var(--jp-icon-yaml);
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /**
   * (DEPRECATED) Support for consuming icons as CSS background images
   */
  
  :root {
    --jp-icon-search-white: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  }
  
  .jp-Icon,
  .jp-MaterialIcon {
    background-position: center;
    background-repeat: no-repeat;
    background-size: 16px;
    min-width: 16px;
    min-height: 16px;
  }
  
  .jp-Icon-cover {
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }
  
  /**
   * (DEPRECATED) Support for specific CSS icon sizes
   */
  
  .jp-Icon-16 {
    background-size: 16px;
    min-width: 16px;
    min-height: 16px;
  }
  
  .jp-Icon-18 {
    background-size: 18px;
    min-width: 18px;
    min-height: 18px;
  }
  
  .jp-Icon-20 {
    background-size: 20px;
    min-width: 20px;
    min-height: 20px;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /**
   * Support for icons as inline SVG HTMLElements
   */
  
  /* recolor the primary elements of an icon */
  .jp-icon0[fill] {
    fill: var(--jp-inverse-layout-color0);
  }
  .jp-icon1[fill] {
    fill: var(--jp-inverse-layout-color1);
  }
  .jp-icon2[fill] {
    fill: var(--jp-inverse-layout-color2);
  }
  .jp-icon3[fill] {
    fill: var(--jp-inverse-layout-color3);
  }
  .jp-icon4[fill] {
    fill: var(--jp-inverse-layout-color4);
  }
  
  .jp-icon0[stroke] {
    stroke: var(--jp-inverse-layout-color0);
  }
  .jp-icon1[stroke] {
    stroke: var(--jp-inverse-layout-color1);
  }
  .jp-icon2[stroke] {
    stroke: var(--jp-inverse-layout-color2);
  }
  .jp-icon3[stroke] {
    stroke: var(--jp-inverse-layout-color3);
  }
  .jp-icon4[stroke] {
    stroke: var(--jp-inverse-layout-color4);
  }
  /* recolor the accent elements of an icon */
  .jp-icon-accent0[fill] {
    fill: var(--jp-layout-color0);
  }
  .jp-icon-accent1[fill] {
    fill: var(--jp-layout-color1);
  }
  .jp-icon-accent2[fill] {
    fill: var(--jp-layout-color2);
  }
  .jp-icon-accent3[fill] {
    fill: var(--jp-layout-color3);
  }
  .jp-icon-accent4[fill] {
    fill: var(--jp-layout-color4);
  }
  
  .jp-icon-accent0[stroke] {
    stroke: var(--jp-layout-color0);
  }
  .jp-icon-accent1[stroke] {
    stroke: var(--jp-layout-color1);
  }
  .jp-icon-accent2[stroke] {
    stroke: var(--jp-layout-color2);
  }
  .jp-icon-accent3[stroke] {
    stroke: var(--jp-layout-color3);
  }
  .jp-icon-accent4[stroke] {
    stroke: var(--jp-layout-color4);
  }
  /* set the color of an icon to transparent */
  .jp-icon-none[fill] {
    fill: none;
  }
  
  .jp-icon-none[stroke] {
    stroke: none;
  }
  /* brand icon colors. Same for light and dark */
  .jp-icon-brand0[fill] {
    fill: var(--jp-brand-color0);
  }
  .jp-icon-brand1[fill] {
    fill: var(--jp-brand-color1);
  }
  .jp-icon-brand2[fill] {
    fill: var(--jp-brand-color2);
  }
  .jp-icon-brand3[fill] {
    fill: var(--jp-brand-color3);
  }
  .jp-icon-brand4[fill] {
    fill: var(--jp-brand-color4);
  }
  
  .jp-icon-brand0[stroke] {
    stroke: var(--jp-brand-color0);
  }
  .jp-icon-brand1[stroke] {
    stroke: var(--jp-brand-color1);
  }
  .jp-icon-brand2[stroke] {
    stroke: var(--jp-brand-color2);
  }
  .jp-icon-brand3[stroke] {
    stroke: var(--jp-brand-color3);
  }
  .jp-icon-brand4[stroke] {
    stroke: var(--jp-brand-color4);
  }
  /* warn icon colors. Same for light and dark */
  .jp-icon-warn0[fill] {
    fill: var(--jp-warn-color0);
  }
  .jp-icon-warn1[fill] {
    fill: var(--jp-warn-color1);
  }
  .jp-icon-warn2[fill] {
    fill: var(--jp-warn-color2);
  }
  .jp-icon-warn3[fill] {
    fill: var(--jp-warn-color3);
  }
  
  .jp-icon-warn0[stroke] {
    stroke: var(--jp-warn-color0);
  }
  .jp-icon-warn1[stroke] {
    stroke: var(--jp-warn-color1);
  }
  .jp-icon-warn2[stroke] {
    stroke: var(--jp-warn-color2);
  }
  .jp-icon-warn3[stroke] {
    stroke: var(--jp-warn-color3);
  }
  /* icon colors that contrast well with each other and most backgrounds */
  .jp-icon-contrast0[fill] {
    fill: var(--jp-icon-contrast-color0);
  }
  .jp-icon-contrast1[fill] {
    fill: var(--jp-icon-contrast-color1);
  }
  .jp-icon-contrast2[fill] {
    fill: var(--jp-icon-contrast-color2);
  }
  .jp-icon-contrast3[fill] {
    fill: var(--jp-icon-contrast-color3);
  }
  
  .jp-icon-contrast0[stroke] {
    stroke: var(--jp-icon-contrast-color0);
  }
  .jp-icon-contrast1[stroke] {
    stroke: var(--jp-icon-contrast-color1);
  }
  .jp-icon-contrast2[stroke] {
    stroke: var(--jp-icon-contrast-color2);
  }
  .jp-icon-contrast3[stroke] {
    stroke: var(--jp-icon-contrast-color3);
  }
  
  /* CSS for icons in selected items in the settings editor */
  #setting-editor .jp-PluginList .jp-mod-selected .jp-icon-selectable[fill] {
    fill: #fff;
  }
  #setting-editor
    .jp-PluginList
    .jp-mod-selected
    .jp-icon-selectable-inverse[fill] {
    fill: var(--jp-brand-color1);
  }
  
  /* CSS for icons in selected filebrowser listing items */
  .jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
    fill: #fff;
  }
  .jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
    fill: var(--jp-brand-color1);
  }
  
  /* CSS for icons in selected tabs in the sidebar tab manager */
  #tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable[fill] {
    fill: #fff;
  }
  
  #tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable-inverse[fill] {
    fill: var(--jp-brand-color1);
  }
  #tab-manager
    .lm-TabBar-tab.jp-mod-active
    .jp-icon-hover
    :hover
    .jp-icon-selectable[fill] {
    fill: var(--jp-brand-color1);
  }
  
  #tab-manager
    .lm-TabBar-tab.jp-mod-active
    .jp-icon-hover
    :hover
    .jp-icon-selectable-inverse[fill] {
    fill: #fff;
  }
  
  /**
   * TODO: come up with non css-hack solution for showing the busy icon on top
   *  of the close icon
   * CSS for complex behavior of close icon of tabs in the sidebar tab manager
   */
  #tab-manager
    .lm-TabBar-tab.jp-mod-dirty
    > .lm-TabBar-tabCloseIcon
    > :not(:hover)
    > .jp-icon3[fill] {
    fill: none;
  }
  #tab-manager
    .lm-TabBar-tab.jp-mod-dirty
    > .lm-TabBar-tabCloseIcon
    > :not(:hover)
    > .jp-icon-busy[fill] {
    fill: var(--jp-inverse-layout-color3);
  }
  
  #tab-manager
    .lm-TabBar-tab.jp-mod-dirty.jp-mod-active
    > .lm-TabBar-tabCloseIcon
    > :not(:hover)
    > .jp-icon-busy[fill] {
    fill: #fff;
  }
  
  /**
  * TODO: come up with non css-hack solution for showing the busy icon on top
  *  of the close icon
  * CSS for complex behavior of close icon of tabs in the main area tabbar
  */
  .lm-DockPanel-tabBar
    .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
    > .lm-TabBar-tabCloseIcon
    > :not(:hover)
    > .jp-icon3[fill] {
    fill: none;
  }
  .lm-DockPanel-tabBar
    .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
    > .lm-TabBar-tabCloseIcon
    > :not(:hover)
    > .jp-icon-busy[fill] {
    fill: var(--jp-inverse-layout-color3);
  }
  
  /* CSS for icons in status bar */
  #jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
    fill: #fff;
  }
  
  #jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
    fill: var(--jp-brand-color1);
  }
  /* special handling for splash icon CSS. While the theme CSS reloads during
     splash, the splash icon can loose theming. To prevent that, we set a
     default for its color variable */
  :root {
    --jp-warn-color0: var(--md-orange-700);
  }
  
  /* not sure what to do with this one, used in filebrowser listing */
  .jp-DragIcon {
    margin-right: 4px;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /**
   * Support for alt colors for icons as inline SVG HTMLElements
   */
  
  /* alt recolor the primary elements of an icon */
  .jp-icon-alt .jp-icon0[fill] {
    fill: var(--jp-layout-color0);
  }
  .jp-icon-alt .jp-icon1[fill] {
    fill: var(--jp-layout-color1);
  }
  .jp-icon-alt .jp-icon2[fill] {
    fill: var(--jp-layout-color2);
  }
  .jp-icon-alt .jp-icon3[fill] {
    fill: var(--jp-layout-color3);
  }
  .jp-icon-alt .jp-icon4[fill] {
    fill: var(--jp-layout-color4);
  }
  
  .jp-icon-alt .jp-icon0[stroke] {
    stroke: var(--jp-layout-color0);
  }
  .jp-icon-alt .jp-icon1[stroke] {
    stroke: var(--jp-layout-color1);
  }
  .jp-icon-alt .jp-icon2[stroke] {
    stroke: var(--jp-layout-color2);
  }
  .jp-icon-alt .jp-icon3[stroke] {
    stroke: var(--jp-layout-color3);
  }
  .jp-icon-alt .jp-icon4[stroke] {
    stroke: var(--jp-layout-color4);
  }
  
  /* alt recolor the accent elements of an icon */
  .jp-icon-alt .jp-icon-accent0[fill] {
    fill: var(--jp-inverse-layout-color0);
  }
  .jp-icon-alt .jp-icon-accent1[fill] {
    fill: var(--jp-inverse-layout-color1);
  }
  .jp-icon-alt .jp-icon-accent2[fill] {
    fill: var(--jp-inverse-layout-color2);
  }
  .jp-icon-alt .jp-icon-accent3[fill] {
    fill: var(--jp-inverse-layout-color3);
  }
  .jp-icon-alt .jp-icon-accent4[fill] {
    fill: var(--jp-inverse-layout-color4);
  }
  
  .jp-icon-alt .jp-icon-accent0[stroke] {
    stroke: var(--jp-inverse-layout-color0);
  }
  .jp-icon-alt .jp-icon-accent1[stroke] {
    stroke: var(--jp-inverse-layout-color1);
  }
  .jp-icon-alt .jp-icon-accent2[stroke] {
    stroke: var(--jp-inverse-layout-color2);
  }
  .jp-icon-alt .jp-icon-accent3[stroke] {
    stroke: var(--jp-inverse-layout-color3);
  }
  .jp-icon-alt .jp-icon-accent4[stroke] {
    stroke: var(--jp-inverse-layout-color4);
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-icon-hoverShow:not(:hover) svg {
    display: none !important;
  }
  
  /**
   * Support for hover colors for icons as inline SVG HTMLElements
   */
  
  /**
   * regular colors
   */
  
  /* recolor the primary elements of an icon */
  .jp-icon-hover :hover .jp-icon0-hover[fill] {
    fill: var(--jp-inverse-layout-color0);
  }
  .jp-icon-hover :hover .jp-icon1-hover[fill] {
    fill: var(--jp-inverse-layout-color1);
  }
  .jp-icon-hover :hover .jp-icon2-hover[fill] {
    fill: var(--jp-inverse-layout-color2);
  }
  .jp-icon-hover :hover .jp-icon3-hover[fill] {
    fill: var(--jp-inverse-layout-color3);
  }
  .jp-icon-hover :hover .jp-icon4-hover[fill] {
    fill: var(--jp-inverse-layout-color4);
  }
  
  .jp-icon-hover :hover .jp-icon0-hover[stroke] {
    stroke: var(--jp-inverse-layout-color0);
  }
  .jp-icon-hover :hover .jp-icon1-hover[stroke] {
    stroke: var(--jp-inverse-layout-color1);
  }
  .jp-icon-hover :hover .jp-icon2-hover[stroke] {
    stroke: var(--jp-inverse-layout-color2);
  }
  .jp-icon-hover :hover .jp-icon3-hover[stroke] {
    stroke: var(--jp-inverse-layout-color3);
  }
  .jp-icon-hover :hover .jp-icon4-hover[stroke] {
    stroke: var(--jp-inverse-layout-color4);
  }
  
  /* recolor the accent elements of an icon */
  .jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
    fill: var(--jp-layout-color0);
  }
  .jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
    fill: var(--jp-layout-color1);
  }
  .jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
    fill: var(--jp-layout-color2);
  }
  .jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
    fill: var(--jp-layout-color3);
  }
  .jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
    fill: var(--jp-layout-color4);
  }
  
  .jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
    stroke: var(--jp-layout-color0);
  }
  .jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
    stroke: var(--jp-layout-color1);
  }
  .jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
    stroke: var(--jp-layout-color2);
  }
  .jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
    stroke: var(--jp-layout-color3);
  }
  .jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
    stroke: var(--jp-layout-color4);
  }
  
  /* set the color of an icon to transparent */
  .jp-icon-hover :hover .jp-icon-none-hover[fill] {
    fill: none;
  }
  
  .jp-icon-hover :hover .jp-icon-none-hover[stroke] {
    stroke: none;
  }
  
  /**
   * inverse colors
   */
  
  /* inverse recolor the primary elements of an icon */
  .jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
    fill: var(--jp-layout-color0);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
    fill: var(--jp-layout-color1);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
    fill: var(--jp-layout-color2);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
    fill: var(--jp-layout-color3);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
    fill: var(--jp-layout-color4);
  }
  
  .jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
    stroke: var(--jp-layout-color0);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
    stroke: var(--jp-layout-color1);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
    stroke: var(--jp-layout-color2);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
    stroke: var(--jp-layout-color3);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
    stroke: var(--jp-layout-color4);
  }
  
  /* inverse recolor the accent elements of an icon */
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
    fill: var(--jp-inverse-layout-color0);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
    fill: var(--jp-inverse-layout-color1);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
    fill: var(--jp-inverse-layout-color2);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
    fill: var(--jp-inverse-layout-color3);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
    fill: var(--jp-inverse-layout-color4);
  }
  
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
    stroke: var(--jp-inverse-layout-color0);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
    stroke: var(--jp-inverse-layout-color1);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
    stroke: var(--jp-inverse-layout-color2);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
    stroke: var(--jp-inverse-layout-color3);
  }
  .jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
    stroke: var(--jp-inverse-layout-color4);
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-switch {
    display: flex;
    align-items: center;
    padding-left: 4px;
    padding-right: 4px;
    font-size: var(--jp-ui-font-size1);
    background-color: transparent;
    color: var(--jp-ui-font-color1);
    border: none;
    height: 20px;
  }
  
  .jp-switch:hover {
    background-color: var(--jp-layout-color2);
  }
  
  .jp-switch-label {
    margin-right: 5px;
  }
  
  .jp-switch-track {
    cursor: pointer;
    background-color: var(--jp-border-color1);
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 34px;
    height: 16px;
    width: 35px;
    position: relative;
  }
  
  .jp-switch-track::before {
    content: '';
    position: absolute;
    height: 10px;
    width: 10px;
    margin: 3px;
    left: 0px;
    background-color: var(--jp-ui-inverse-font-color1);
    -webkit-transition: 0.4s;
    transition: 0.4s;
    border-radius: 50%;
  }
  
  .jp-switch[aria-checked='true'] .jp-switch-track {
    background-color: var(--jp-warn-color0);
  }
  
  .jp-switch[aria-checked='true'] .jp-switch-track::before {
    /* track width (35) - margins (3 + 3) - thumb width (10) */
    left: 19px;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /* Sibling imports */
  
  /* Override Blueprint's _reset.scss styles */
  html {
    box-sizing: unset;
  }
  
  *,
  *::before,
  *::after {
    box-sizing: unset;
  }
  
  body {
    color: unset;
    font-family: var(--jp-ui-font-family);
  }
  
  p {
    margin-top: unset;
    margin-bottom: unset;
  }
  
  small {
    font-size: unset;
  }
  
  strong {
    font-weight: unset;
  }
  
  /* Override Blueprint's _typography.scss styles */
  a {
    text-decoration: unset;
    color: unset;
  }
  a:hover {
    text-decoration: unset;
    color: unset;
  }
  
  /* Override Blueprint's _accessibility.scss styles */
  :focus {
    outline: unset;
    outline-offset: unset;
    -moz-outline-radius: unset;
  }
  
  /* Styles for ui-components */
  .jp-Button {
    border-radius: var(--jp-border-radius);
    padding: 0px 12px;
    font-size: var(--jp-ui-font-size1);
  }
  
  /* Use our own theme for hover styles */
  button.jp-Button.bp3-button.bp3-minimal:hover {
    background-color: var(--jp-layout-color2);
  }
  .jp-Button.minimal {
    color: unset !important;
  }
  
  .jp-Button.jp-ToolbarButtonComponent {
    text-transform: none;
  }
  
  .jp-InputGroup input {
    box-sizing: border-box;
    border-radius: 0;
    background-color: transparent;
    color: var(--jp-ui-font-color0);
    box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  }
  
  .jp-InputGroup input:focus {
    box-shadow: inset 0 0 0 var(--jp-border-width)
        var(--jp-input-active-box-shadow-color),
      inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
  }
  
  .jp-InputGroup input::placeholder,
  input::placeholder {
    color: var(--jp-ui-font-color3);
  }
  
  .jp-BPIcon {
    display: inline-block;
    vertical-align: middle;
    margin: auto;
  }
  
  /* Stop blueprint futzing with our icon fills */
  .bp3-icon.jp-BPIcon > svg:not([fill]) {
    fill: var(--jp-inverse-layout-color3);
  }
  
  .jp-InputGroupAction {
    padding: 6px;
  }
  
  .jp-HTMLSelect.jp-DefaultStyle select {
    background-color: initial;
    border: none;
    border-radius: 0;
    box-shadow: none;
    color: var(--jp-ui-font-color0);
    display: block;
    font-size: var(--jp-ui-font-size1);
    height: 24px;
    line-height: 14px;
    padding: 0 25px 0 10px;
    text-align: left;
    -moz-appearance: none;
    -webkit-appearance: none;
  }
  
  /* Use our own theme for hover and option styles */
  .jp-HTMLSelect.jp-DefaultStyle select:hover,
  .jp-HTMLSelect.jp-DefaultStyle select > option {
    background-color: var(--jp-layout-color2);
    color: var(--jp-ui-font-color0);
  }
  select {
    box-sizing: border-box;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-Collapse {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    border-top: 1px solid var(--jp-border-color2);
    border-bottom: 1px solid var(--jp-border-color2);
  }
  
  .jp-Collapse-header {
    padding: 1px 12px;
    color: var(--jp-ui-font-color1);
    background-color: var(--jp-layout-color1);
    font-size: var(--jp-ui-font-size2);
  }
  
  .jp-Collapse-header:hover {
    background-color: var(--jp-layout-color2);
  }
  
  .jp-Collapse-contents {
    padding: 0px 12px 0px 12px;
    background-color: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
    overflow: auto;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Variables
  |----------------------------------------------------------------------------*/
  
  :root {
    --jp-private-commandpalette-search-height: 28px;
  }
  
  /*-----------------------------------------------------------------------------
  | Overall styles
  |----------------------------------------------------------------------------*/
  
  .lm-CommandPalette {
    padding-bottom: 0px;
    color: var(--jp-ui-font-color1);
    background: var(--jp-layout-color1);
    /* This is needed so that all font sizing of children done in ems is
     * relative to this base size */
    font-size: var(--jp-ui-font-size1);
  }
  
  /*-----------------------------------------------------------------------------
  | Modal variant
  |----------------------------------------------------------------------------*/
  
  .jp-ModalCommandPalette {
    position: absolute;
    z-index: 10000;
    top: 38px;
    left: 30%;
    margin: 0;
    padding: 4px;
    width: 40%;
    box-shadow: var(--jp-elevation-z4);
    border-radius: 4px;
    background: var(--jp-layout-color0);
  }
  
  .jp-ModalCommandPalette .lm-CommandPalette {
    max-height: 40vh;
  }
  
  .jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
    display: none;
  }
  
  .jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
    display: none;
  }
  
  .jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
    margin-left: 4px;
    margin-right: 4px;
  }
  
  .jp-ModalCommandPalette
    .lm-CommandPalette
    .lm-CommandPalette-item.lm-mod-disabled {
    display: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Search
  |----------------------------------------------------------------------------*/
  
  .lm-CommandPalette-search {
    padding: 4px;
    background-color: var(--jp-layout-color1);
    z-index: 2;
  }
  
  .lm-CommandPalette-wrapper {
    overflow: overlay;
    padding: 0px 9px;
    background-color: var(--jp-input-active-background);
    height: 30px;
    box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  }
  
  .lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
    box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
      inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
  }
  
  .lm-CommandPalette-wrapper::after {
    content: ' ';
    color: white;
    background-color: var(--jp-brand-color1);
    position: absolute;
    top: 4px;
    right: 4px;
    height: 30px;
    width: 10px;
    padding: 0px 10px;
    background-image: var(--jp-icon-search-white);
    background-size: 20px;
    background-repeat: no-repeat;
    background-position: center;
  }
  
  .lm-CommandPalette-input {
    background: transparent;
    width: calc(100% - 18px);
    float: left;
    border: none;
    outline: none;
    font-size: var(--jp-ui-font-size1);
    color: var(--jp-ui-font-color0);
    line-height: var(--jp-private-commandpalette-search-height);
  }
  
  .lm-CommandPalette-input::-webkit-input-placeholder,
  .lm-CommandPalette-input::-moz-placeholder,
  .lm-CommandPalette-input:-ms-input-placeholder {
    color: var(--jp-ui-font-color3);
    font-size: var(--jp-ui-font-size1);
  }
  
  /*-----------------------------------------------------------------------------
  | Results
  |----------------------------------------------------------------------------*/
  
  .lm-CommandPalette-header:first-child {
    margin-top: 0px;
  }
  
  .lm-CommandPalette-header {
    border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
    color: var(--jp-ui-font-color1);
    cursor: pointer;
    display: flex;
    font-size: var(--jp-ui-font-size0);
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 8px;
    padding: 8px 0 8px 12px;
    text-transform: uppercase;
  }
  
  .lm-CommandPalette-header.lm-mod-active {
    background: var(--jp-layout-color2);
  }
  
  .lm-CommandPalette-header > mark {
    background-color: transparent;
    font-weight: bold;
    color: var(--jp-ui-font-color1);
  }
  
  .lm-CommandPalette-item {
    padding: 4px 12px 4px 4px;
    color: var(--jp-ui-font-color1);
    font-size: var(--jp-ui-font-size1);
    font-weight: 400;
    display: flex;
  }
  
  .lm-CommandPalette-item.lm-mod-disabled {
    color: var(--jp-ui-font-color3);
  }
  
  .lm-CommandPalette-item.lm-mod-active {
    background: var(--jp-layout-color3);
  }
  
  .lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
    background: var(--jp-layout-color4);
  }
  
  .lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
    background: var(--jp-layout-color2);
  }
  
  .lm-CommandPalette-itemContent {
    overflow: hidden;
  }
  
  .lm-CommandPalette-itemLabel > mark {
    color: var(--jp-ui-font-color0);
    background-color: transparent;
    font-weight: bold;
  }
  
  .lm-CommandPalette-item.lm-mod-disabled mark {
    color: var(--jp-ui-font-color3);
  }
  
  .lm-CommandPalette-item .lm-CommandPalette-itemIcon {
    margin: 0 4px 0 0;
    position: relative;
    width: 16px;
    top: 2px;
    flex: 0 0 auto;
  }
  
  .lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
    opacity: 0.4;
  }
  
  .lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
    flex: 0 0 auto;
  }
  
  .lm-CommandPalette-itemCaption {
    display: none;
  }
  
  .lm-CommandPalette-content {
    background-color: var(--jp-layout-color1);
  }
  
  .lm-CommandPalette-content:empty:after {
    content: 'No results';
    margin: auto;
    margin-top: 20px;
    width: 100px;
    display: block;
    font-size: var(--jp-ui-font-size2);
    font-family: var(--jp-ui-font-family);
    font-weight: lighter;
  }
  
  .lm-CommandPalette-emptyMessage {
    text-align: center;
    margin-top: 24px;
    line-height: 1.32;
    padding: 0px 8px;
    color: var(--jp-content-font-color3);
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2014-2017, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-Dialog {
    position: absolute;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    top: 0px;
    left: 0px;
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    background: var(--jp-dialog-background);
  }
  
  .jp-Dialog-content {
    display: flex;
    flex-direction: column;
    margin-left: auto;
    margin-right: auto;
    background: var(--jp-layout-color1);
    padding: 24px;
    padding-bottom: 12px;
    min-width: 300px;
    min-height: 150px;
    max-width: 1000px;
    max-height: 500px;
    box-sizing: border-box;
    box-shadow: var(--jp-elevation-z20);
    word-wrap: break-word;
    border-radius: var(--jp-border-radius);
    /* This is needed so that all font sizing of children done in ems is
     * relative to this base size */
    font-size: var(--jp-ui-font-size1);
    color: var(--jp-ui-font-color1);
    resize: both;
  }
  
  .jp-Dialog-button {
    overflow: visible;
  }
  
  button.jp-Dialog-button:focus {
    outline: 1px solid var(--jp-brand-color1);
    outline-offset: 4px;
    -moz-outline-radius: 0px;
  }
  
  button.jp-Dialog-button:focus::-moz-focus-inner {
    border: 0;
  }
  
  button.jp-Dialog-close-button {
    padding: 0;
    height: 100%;
    min-width: unset;
    min-height: unset;
  }
  
  .jp-Dialog-header {
    display: flex;
    justify-content: space-between;
    flex: 0 0 auto;
    padding-bottom: 12px;
    font-size: var(--jp-ui-font-size3);
    font-weight: 400;
    color: var(--jp-ui-font-color0);
  }
  
  .jp-Dialog-body {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    font-size: var(--jp-ui-font-size1);
    background: var(--jp-layout-color1);
    overflow: auto;
  }
  
  .jp-Dialog-footer {
    display: flex;
    flex-direction: row;
    justify-content: flex-end;
    flex: 0 0 auto;
    margin-left: -12px;
    margin-right: -12px;
    padding: 12px;
  }
  
  .jp-Dialog-title {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  
  .jp-Dialog-body > .jp-select-wrapper {
    width: 100%;
  }
  
  .jp-Dialog-body > button {
    padding: 0px 16px;
  }
  
  .jp-Dialog-body > label {
    line-height: 1.4;
    color: var(--jp-ui-font-color0);
  }
  
  .jp-Dialog-button.jp-mod-styled:not(:last-child) {
    margin-right: 12px;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2014-2016, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-HoverBox {
    position: fixed;
  }
  
  .jp-HoverBox.jp-mod-outofview {
    display: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-IFrame {
    width: 100%;
    height: 100%;
  }
  
  .jp-IFrame > iframe {
    border: none;
  }
  
  /*
  When drag events occur, `p-mod-override-cursor` is added to the body.
  Because iframes steal all cursor events, the following two rules are necessary
  to suppress pointer events while resize drags are occurring. There may be a
  better solution to this problem.
  */
  body.lm-mod-override-cursor .jp-IFrame {
    position: relative;
  }
  
  body.lm-mod-override-cursor .jp-IFrame:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: transparent;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2014-2016, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-MainAreaWidget > :focus {
    outline: none;
  }
  
  /**
   * google-material-color v1.2.6
   * https://github.com/danlevan/google-material-color
   */
  :root {
    --md-red-50: #ffebee;
    --md-red-100: #ffcdd2;
    --md-red-200: #ef9a9a;
    --md-red-300: #e57373;
    --md-red-400: #ef5350;
    --md-red-500: #f44336;
    --md-red-600: #e53935;
    --md-red-700: #d32f2f;
    --md-red-800: #c62828;
    --md-red-900: #b71c1c;
    --md-red-A100: #ff8a80;
    --md-red-A200: #ff5252;
    --md-red-A400: #ff1744;
    --md-red-A700: #d50000;
  
    --md-pink-50: #fce4ec;
    --md-pink-100: #f8bbd0;
    --md-pink-200: #f48fb1;
    --md-pink-300: #f06292;
    --md-pink-400: #ec407a;
    --md-pink-500: #e91e63;
    --md-pink-600: #d81b60;
    --md-pink-700: #c2185b;
    --md-pink-800: #ad1457;
    --md-pink-900: #880e4f;
    --md-pink-A100: #ff80ab;
    --md-pink-A200: #ff4081;
    --md-pink-A400: #f50057;
    --md-pink-A700: #c51162;
  
    --md-purple-50: #f3e5f5;
    --md-purple-100: #e1bee7;
    --md-purple-200: #ce93d8;
    --md-purple-300: #ba68c8;
    --md-purple-400: #ab47bc;
    --md-purple-500: #9c27b0;
    --md-purple-600: #8e24aa;
    --md-purple-700: #7b1fa2;
    --md-purple-800: #6a1b9a;
    --md-purple-900: #4a148c;
    --md-purple-A100: #ea80fc;
    --md-purple-A200: #e040fb;
    --md-purple-A400: #d500f9;
    --md-purple-A700: #aa00ff;
  
    --md-deep-purple-50: #ede7f6;
    --md-deep-purple-100: #d1c4e9;
    --md-deep-purple-200: #b39ddb;
    --md-deep-purple-300: #9575cd;
    --md-deep-purple-400: #7e57c2;
    --md-deep-purple-500: #673ab7;
    --md-deep-purple-600: #5e35b1;
    --md-deep-purple-700: #512da8;
    --md-deep-purple-800: #4527a0;
    --md-deep-purple-900: #311b92;
    --md-deep-purple-A100: #b388ff;
    --md-deep-purple-A200: #7c4dff;
    --md-deep-purple-A400: #651fff;
    --md-deep-purple-A700: #6200ea;
  
    --md-indigo-50: #e8eaf6;
    --md-indigo-100: #c5cae9;
    --md-indigo-200: #9fa8da;
    --md-indigo-300: #7986cb;
    --md-indigo-400: #5c6bc0;
    --md-indigo-500: #3f51b5;
    --md-indigo-600: #3949ab;
    --md-indigo-700: #303f9f;
    --md-indigo-800: #283593;
    --md-indigo-900: #1a237e;
    --md-indigo-A100: #8c9eff;
    --md-indigo-A200: #536dfe;
    --md-indigo-A400: #3d5afe;
    --md-indigo-A700: #304ffe;
  
    --md-blue-50: #e3f2fd;
    --md-blue-100: #bbdefb;
    --md-blue-200: #90caf9;
    --md-blue-300: #64b5f6;
    --md-blue-400: #42a5f5;
    --md-blue-500: #2196f3;
    --md-blue-600: #1e88e5;
    --md-blue-700: #1976d2;
    --md-blue-800: #1565c0;
    --md-blue-900: #0d47a1;
    --md-blue-A100: #82b1ff;
    --md-blue-A200: #448aff;
    --md-blue-A400: #2979ff;
    --md-blue-A700: #2962ff;
  
    --md-light-blue-50: #e1f5fe;
    --md-light-blue-100: #b3e5fc;
    --md-light-blue-200: #81d4fa;
    --md-light-blue-300: #4fc3f7;
    --md-light-blue-400: #29b6f6;
    --md-light-blue-500: #03a9f4;
    --md-light-blue-600: #039be5;
    --md-light-blue-700: #0288d1;
    --md-light-blue-800: #0277bd;
    --md-light-blue-900: #01579b;
    --md-light-blue-A100: #80d8ff;
    --md-light-blue-A200: #40c4ff;
    --md-light-blue-A400: #00b0ff;
    --md-light-blue-A700: #0091ea;
  
    --md-cyan-50: #e0f7fa;
    --md-cyan-100: #b2ebf2;
    --md-cyan-200: #80deea;
    --md-cyan-300: #4dd0e1;
    --md-cyan-400: #26c6da;
    --md-cyan-500: #00bcd4;
    --md-cyan-600: #00acc1;
    --md-cyan-700: #0097a7;
    --md-cyan-800: #00838f;
    --md-cyan-900: #006064;
    --md-cyan-A100: #84ffff;
    --md-cyan-A200: #18ffff;
    --md-cyan-A400: #00e5ff;
    --md-cyan-A700: #00b8d4;
  
    --md-teal-50: #e0f2f1;
    --md-teal-100: #b2dfdb;
    --md-teal-200: #80cbc4;
    --md-teal-300: #4db6ac;
    --md-teal-400: #26a69a;
    --md-teal-500: #009688;
    --md-teal-600: #00897b;
    --md-teal-700: #00796b;
    --md-teal-800: #00695c;
    --md-teal-900: #004d40;
    --md-teal-A100: #a7ffeb;
    --md-teal-A200: #64ffda;
    --md-teal-A400: #1de9b6;
    --md-teal-A700: #00bfa5;
  
    --md-green-50: #e8f5e9;
    --md-green-100: #c8e6c9;
    --md-green-200: #a5d6a7;
    --md-green-300: #81c784;
    --md-green-400: #66bb6a;
    --md-green-500: #4caf50;
    --md-green-600: #43a047;
    --md-green-700: #388e3c;
    --md-green-800: #2e7d32;
    --md-green-900: #1b5e20;
    --md-green-A100: #b9f6ca;
    --md-green-A200: #69f0ae;
    --md-green-A400: #00e676;
    --md-green-A700: #00c853;
  
    --md-light-green-50: #f1f8e9;
    --md-light-green-100: #dcedc8;
    --md-light-green-200: #c5e1a5;
    --md-light-green-300: #aed581;
    --md-light-green-400: #9ccc65;
    --md-light-green-500: #8bc34a;
    --md-light-green-600: #7cb342;
    --md-light-green-700: #689f38;
    --md-light-green-800: #558b2f;
    --md-light-green-900: #33691e;
    --md-light-green-A100: #ccff90;
    --md-light-green-A200: #b2ff59;
    --md-light-green-A400: #76ff03;
    --md-light-green-A700: #64dd17;
  
    --md-lime-50: #f9fbe7;
    --md-lime-100: #f0f4c3;
    --md-lime-200: #e6ee9c;
    --md-lime-300: #dce775;
    --md-lime-400: #d4e157;
    --md-lime-500: #cddc39;
    --md-lime-600: #c0ca33;
    --md-lime-700: #afb42b;
    --md-lime-800: #9e9d24;
    --md-lime-900: #827717;
    --md-lime-A100: #f4ff81;
    --md-lime-A200: #eeff41;
    --md-lime-A400: #c6ff00;
    --md-lime-A700: #aeea00;
  
    --md-yellow-50: #fffde7;
    --md-yellow-100: #fff9c4;
    --md-yellow-200: #fff59d;
    --md-yellow-300: #fff176;
    --md-yellow-400: #ffee58;
    --md-yellow-500: #ffeb3b;
    --md-yellow-600: #fdd835;
    --md-yellow-700: #fbc02d;
    --md-yellow-800: #f9a825;
    --md-yellow-900: #f57f17;
    --md-yellow-A100: #ffff8d;
    --md-yellow-A200: #ffff00;
    --md-yellow-A400: #ffea00;
    --md-yellow-A700: #ffd600;
  
    --md-amber-50: #fff8e1;
    --md-amber-100: #ffecb3;
    --md-amber-200: #ffe082;
    --md-amber-300: #ffd54f;
    --md-amber-400: #ffca28;
    --md-amber-500: #ffc107;
    --md-amber-600: #ffb300;
    --md-amber-700: #ffa000;
    --md-amber-800: #ff8f00;
    --md-amber-900: #ff6f00;
    --md-amber-A100: #ffe57f;
    --md-amber-A200: #ffd740;
    --md-amber-A400: #ffc400;
    --md-amber-A700: #ffab00;
  
    --md-orange-50: #fff3e0;
    --md-orange-100: #ffe0b2;
    --md-orange-200: #ffcc80;
    --md-orange-300: #ffb74d;
    --md-orange-400: #ffa726;
    --md-orange-500: #ff9800;
    --md-orange-600: #fb8c00;
    --md-orange-700: #f57c00;
    --md-orange-800: #ef6c00;
    --md-orange-900: #e65100;
    --md-orange-A100: #ffd180;
    --md-orange-A200: #ffab40;
    --md-orange-A400: #ff9100;
    --md-orange-A700: #ff6d00;
  
    --md-deep-orange-50: #fbe9e7;
    --md-deep-orange-100: #ffccbc;
    --md-deep-orange-200: #ffab91;
    --md-deep-orange-300: #ff8a65;
    --md-deep-orange-400: #ff7043;
    --md-deep-orange-500: #ff5722;
    --md-deep-orange-600: #f4511e;
    --md-deep-orange-700: #e64a19;
    --md-deep-orange-800: #d84315;
    --md-deep-orange-900: #bf360c;
    --md-deep-orange-A100: #ff9e80;
    --md-deep-orange-A200: #ff6e40;
    --md-deep-orange-A400: #ff3d00;
    --md-deep-orange-A700: #dd2c00;
  
    --md-brown-50: #efebe9;
    --md-brown-100: #d7ccc8;
    --md-brown-200: #bcaaa4;
    --md-brown-300: #a1887f;
    --md-brown-400: #8d6e63;
    --md-brown-500: #795548;
    --md-brown-600: #6d4c41;
    --md-brown-700: #5d4037;
    --md-brown-800: #4e342e;
    --md-brown-900: #3e2723;
  
    --md-grey-50: #fafafa;
    --md-grey-100: #f5f5f5;
    --md-grey-200: #eeeeee;
    --md-grey-300: #e0e0e0;
    --md-grey-400: #bdbdbd;
    --md-grey-500: #9e9e9e;
    --md-grey-600: #757575;
    --md-grey-700: #616161;
    --md-grey-800: #424242;
    --md-grey-900: #212121;
  
    --md-blue-grey-50: #eceff1;
    --md-blue-grey-100: #cfd8dc;
    --md-blue-grey-200: #b0bec5;
    --md-blue-grey-300: #90a4ae;
    --md-blue-grey-400: #78909c;
    --md-blue-grey-500: #607d8b;
    --md-blue-grey-600: #546e7a;
    --md-blue-grey-700: #455a64;
    --md-blue-grey-800: #37474f;
    --md-blue-grey-900: #263238;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2017, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-Spinner {
    position: absolute;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: var(--jp-layout-color0);
    outline: none;
  }
  
  .jp-SpinnerContent {
    font-size: 10px;
    margin: 50px auto;
    text-indent: -9999em;
    width: 3em;
    height: 3em;
    border-radius: 50%;
    background: var(--jp-brand-color3);
    background: linear-gradient(
      to right,
      #f37626 10%,
      rgba(255, 255, 255, 0) 42%
    );
    position: relative;
    animation: load3 1s infinite linear, fadeIn 1s;
  }
  
  .jp-SpinnerContent:before {
    width: 50%;
    height: 50%;
    background: #f37626;
    border-radius: 100% 0 0 0;
    position: absolute;
    top: 0;
    left: 0;
    content: '';
  }
  
  .jp-SpinnerContent:after {
    background: var(--jp-layout-color0);
    width: 75%;
    height: 75%;
    border-radius: 50%;
    content: '';
    margin: auto;
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    right: 0;
  }
  
  @keyframes fadeIn {
    0% {
      opacity: 0;
    }
    100% {
      opacity: 1;
    }
  }
  
  @keyframes load3 {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2014-2017, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  button.jp-mod-styled {
    font-size: var(--jp-ui-font-size1);
    color: var(--jp-ui-font-color0);
    border: none;
    box-sizing: border-box;
    text-align: center;
    line-height: 32px;
    height: 32px;
    padding: 0px 12px;
    letter-spacing: 0.8px;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
  }
  
  input.jp-mod-styled {
    background: var(--jp-input-background);
    height: 28px;
    box-sizing: border-box;
    border: var(--jp-border-width) solid var(--jp-border-color1);
    padding-left: 7px;
    padding-right: 7px;
    font-size: var(--jp-ui-font-size2);
    color: var(--jp-ui-font-color0);
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
  }
  
  input.jp-mod-styled:focus {
    border: var(--jp-border-width) solid var(--md-blue-500);
    box-shadow: inset 0 0 4px var(--md-blue-300);
  }
  
  .jp-select-wrapper {
    display: flex;
    position: relative;
    flex-direction: column;
    padding: 1px;
    background-color: var(--jp-layout-color1);
    height: 28px;
    box-sizing: border-box;
    margin-bottom: 12px;
  }
  
  .jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
    border: var(--jp-border-width) solid var(--jp-input-active-border-color);
    box-shadow: var(--jp-input-box-shadow);
    background-color: var(--jp-input-active-background);
  }
  
  select.jp-mod-styled:hover {
    background-color: var(--jp-layout-color1);
    cursor: pointer;
    color: var(--jp-ui-font-color0);
    background-color: var(--jp-input-hover-background);
    box-shadow: inset 0 0px 1px rgba(0, 0, 0, 0.5);
  }
  
  select.jp-mod-styled {
    flex: 1 1 auto;
    height: 32px;
    width: 100%;
    font-size: var(--jp-ui-font-size2);
    background: var(--jp-input-background);
    color: var(--jp-ui-font-color0);
    padding: 0 25px 0 8px;
    border: var(--jp-border-width) solid var(--jp-input-border-color);
    border-radius: 0px;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2014-2016, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  :root {
    --jp-private-toolbar-height: calc(
      28px + var(--jp-border-width)
    ); /* leave 28px for content */
  }
  
  .jp-Toolbar {
    color: var(--jp-ui-font-color1);
    flex: 0 0 auto;
    display: flex;
    flex-direction: row;
    border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
    box-shadow: var(--jp-toolbar-box-shadow);
    background: var(--jp-toolbar-background);
    min-height: var(--jp-toolbar-micro-height);
    padding: 2px;
    z-index: 1;
    overflow-x: hidden;
  }
  
  .jp-Toolbar:hover {
    overflow-x: auto;
  }
  
  /* Toolbar items */
  
  .jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
    flex-grow: 1;
    flex-shrink: 1;
  }
  
  .jp-Toolbar-item.jp-Toolbar-kernelStatus {
    display: inline-block;
    width: 32px;
    background-repeat: no-repeat;
    background-position: center;
    background-size: 16px;
  }
  
  .jp-Toolbar > .jp-Toolbar-item {
    flex: 0 0 auto;
    display: flex;
    padding-left: 1px;
    padding-right: 1px;
    font-size: var(--jp-ui-font-size1);
    line-height: var(--jp-private-toolbar-height);
    height: 100%;
  }
  
  /* Toolbar buttons */
  
  /* This is the div we use to wrap the react component into a Widget */
  div.jp-ToolbarButton {
    color: transparent;
    border: none;
    box-sizing: border-box;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    padding: 0px;
    margin: 0px;
  }
  
  button.jp-ToolbarButtonComponent {
    background: var(--jp-layout-color1);
    border: none;
    box-sizing: border-box;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    padding: 0px 6px;
    margin: 0px;
    height: 24px;
    border-radius: var(--jp-border-radius);
    display: flex;
    align-items: center;
    text-align: center;
    font-size: 14px;
    min-width: unset;
    min-height: unset;
  }
  
  button.jp-ToolbarButtonComponent:disabled {
    opacity: 0.4;
  }
  
  button.jp-ToolbarButtonComponent span {
    padding: 0px;
    flex: 0 0 auto;
  }
  
  button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
    font-size: var(--jp-ui-font-size1);
    line-height: 100%;
    padding-left: 2px;
    color: var(--jp-ui-font-color1);
  }
  
  #jp-main-dock-panel[data-mode='single-document']
    .jp-MainAreaWidget
    > .jp-Toolbar.jp-Toolbar-micro {
    padding: 0;
    min-height: 0;
  }
  
  #jp-main-dock-panel[data-mode='single-document']
    .jp-MainAreaWidget
    > .jp-Toolbar {
    border: none;
    box-shadow: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2014-2017, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Copyright (c) 2014-2017, PhosphorJS Contributors
  |
  | Distributed under the terms of the BSD 3-Clause License.
  |
  | The full license is in the file LICENSE, distributed with this software.
  |----------------------------------------------------------------------------*/
  
  
  /* <DEPRECATED> */ body.p-mod-override-cursor *, /* </DEPRECATED> */
  body.lm-mod-override-cursor * {
    cursor: inherit !important;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) 2014-2016, Jupyter Development Team.
  |
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-JSONEditor {
    display: flex;
    flex-direction: column;
    width: 100%;
  }
  
  .jp-JSONEditor-host {
    flex: 1 1 auto;
    border: var(--jp-border-width) solid var(--jp-input-border-color);
    border-radius: 0px;
    background: var(--jp-layout-color0);
    min-height: 50px;
    padding: 1px;
  }
  
  .jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
    border-color: red;
    outline-color: red;
  }
  
  .jp-JSONEditor-header {
    display: flex;
    flex: 1 0 auto;
    padding: 0 0 0 12px;
  }
  
  .jp-JSONEditor-header label {
    flex: 0 0 auto;
  }
  
  .jp-JSONEditor-commitButton {
    height: 16px;
    width: 16px;
    background-size: 18px;
    background-repeat: no-repeat;
    background-position: center;
  }
  
  .jp-JSONEditor-host.jp-mod-focused {
    background-color: var(--jp-input-active-background);
    border: 1px solid var(--jp-input-active-border-color);
    box-shadow: var(--jp-input-box-shadow);
  }
  
  .jp-Editor.jp-mod-dropTarget {
    border: var(--jp-border-width) solid var(--jp-input-active-border-color);
    box-shadow: var(--jp-input-box-shadow);
  }
  
  /* BASICS */
  
  .CodeMirror {
    /* Set height, width, borders, and global font properties here */
    font-family: monospace;
    height: 300px;
    color: black;
    direction: ltr;
  }
  
  /* PADDING */
  
  .CodeMirror-lines {
    padding: 4px 0; /* Vertical padding around content */
  }
  .CodeMirror pre.CodeMirror-line,
  .CodeMirror pre.CodeMirror-line-like {
    padding: 0 4px; /* Horizontal padding of content */
  }
  
  .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
    background-color: white; /* The little square between H and V scrollbars */
  }
  
  /* GUTTER */
  
  .CodeMirror-gutters {
    border-right: 1px solid #ddd;
    background-color: #f7f7f7;
    white-space: nowrap;
  }
  .CodeMirror-linenumbers {}
  .CodeMirror-linenumber {
    padding: 0 3px 0 5px;
    min-width: 20px;
    text-align: right;
    color: #999;
    white-space: nowrap;
  }
  
  .CodeMirror-guttermarker { color: black; }
  .CodeMirror-guttermarker-subtle { color: #999; }
  
  /* CURSOR */
  
  .CodeMirror-cursor {
    border-left: 1px solid black;
    border-right: none;
    width: 0;
  }
  /* Shown when moving in bi-directional text */
  .CodeMirror div.CodeMirror-secondarycursor {
    border-left: 1px solid silver;
  }
  .cm-fat-cursor .CodeMirror-cursor {
    width: auto;
    border: 0 !important;
    background: #7e7;
  }
  .cm-fat-cursor div.CodeMirror-cursors {
    z-index: 1;
  }
  .cm-fat-cursor-mark {
    background-color: rgba(20, 255, 20, 0.5);
    -webkit-animation: blink 1.06s steps(1) infinite;
    -moz-animation: blink 1.06s steps(1) infinite;
    animation: blink 1.06s steps(1) infinite;
  }
  .cm-animate-fat-cursor {
    width: auto;
    border: 0;
    -webkit-animation: blink 1.06s steps(1) infinite;
    -moz-animation: blink 1.06s steps(1) infinite;
    animation: blink 1.06s steps(1) infinite;
    background-color: #7e7;
  }
  @-moz-keyframes blink {
    0% {}
    50% { background-color: transparent; }
    100% {}
  }
  @-webkit-keyframes blink {
    0% {}
    50% { background-color: transparent; }
    100% {}
  }
  @keyframes blink {
    0% {}
    50% { background-color: transparent; }
    100% {}
  }
  
  /* Can style cursor different in overwrite (non-insert) mode */
  .CodeMirror-overwrite .CodeMirror-cursor {}
  
  .cm-tab { display: inline-block; text-decoration: inherit; }
  
  .CodeMirror-rulers {
    position: absolute;
    left: 0; right: 0; top: -50px; bottom: 0;
    overflow: hidden;
  }
  .CodeMirror-ruler {
    border-left: 1px solid #ccc;
    top: 0; bottom: 0;
    position: absolute;
  }
  
  /* DEFAULT THEME */
  
  .cm-s-default .cm-header {color: blue;}
  .cm-s-default .cm-quote {color: #090;}
  .cm-negative {color: #d44;}
  .cm-positive {color: #292;}
  .cm-header, .cm-strong {font-weight: bold;}
  .cm-em {font-style: italic;}
  .cm-link {text-decoration: underline;}
  .cm-strikethrough {text-decoration: line-through;}
  
  .cm-s-default .cm-keyword {color: #708;}
  .cm-s-default .cm-atom {color: #219;}
  .cm-s-default .cm-number {color: #164;}
  .cm-s-default .cm-def {color: #00f;}
  .cm-s-default .cm-variable,
  .cm-s-default .cm-punctuation,
  .cm-s-default .cm-property,
  .cm-s-default .cm-operator {}
  .cm-s-default .cm-variable-2 {color: #05a;}
  .cm-s-default .cm-variable-3, .cm-s-default .cm-type {color: #085;}
  .cm-s-default .cm-comment {color: #a50;}
  .cm-s-default .cm-string {color: #a11;}
  .cm-s-default .cm-string-2 {color: #f50;}
  .cm-s-default .cm-meta {color: #555;}
  .cm-s-default .cm-qualifier {color: #555;}
  .cm-s-default .cm-builtin {color: #30a;}
  .cm-s-default .cm-bracket {color: #997;}
  .cm-s-default .cm-tag {color: #170;}
  .cm-s-default .cm-attribute {color: #00c;}
  .cm-s-default .cm-hr {color: #999;}
  .cm-s-default .cm-link {color: #00c;}
  
  .cm-s-default .cm-error {color: #f00;}
  .cm-invalidchar {color: #f00;}
  
  .CodeMirror-composing { border-bottom: 2px solid; }
  
  /* Default styles for common addons */
  
  div.CodeMirror span.CodeMirror-matchingbracket {color: #0b0;}
  div.CodeMirror span.CodeMirror-nonmatchingbracket {color: #a22;}
  .CodeMirror-matchingtag { background: rgba(255, 150, 0, .3); }
  .CodeMirror-activeline-background {background: #e8f2ff;}
  
  /* STOP */
  
  /* The rest of this file contains styles related to the mechanics of
     the editor. You probably shouldn't touch them. */
  
  .CodeMirror {
    position: relative;
    overflow: hidden;
    background: white;
  }
  
  .CodeMirror-scroll {
    overflow: scroll !important; /* Things will break if this is overridden */
    /* 50px is the magic margin used to hide the element's real scrollbars */
    /* See overflow: hidden in .CodeMirror */
    margin-bottom: -50px; margin-right: -50px;
    padding-bottom: 50px;
    height: 100%;
    outline: none; /* Prevent dragging from highlighting the element */
    position: relative;
  }
  .CodeMirror-sizer {
    position: relative;
    border-right: 50px solid transparent;
  }
  
  /* The fake, visible scrollbars. Used to force redraw during scrolling
     before actual scrolling happens, thus preventing shaking and
     flickering artifacts. */
  .CodeMirror-vscrollbar, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
    position: absolute;
    z-index: 6;
    display: none;
  }
  .CodeMirror-vscrollbar {
    right: 0; top: 0;
    overflow-x: hidden;
    overflow-y: scroll;
  }
  .CodeMirror-hscrollbar {
    bottom: 0; left: 0;
    overflow-y: hidden;
    overflow-x: scroll;
  }
  .CodeMirror-scrollbar-filler {
    right: 0; bottom: 0;
  }
  .CodeMirror-gutter-filler {
    left: 0; bottom: 0;
  }
  
  .CodeMirror-gutters {
    position: absolute; left: 0; top: 0;
    min-height: 100%;
    z-index: 3;
  }
  .CodeMirror-gutter {
    white-space: normal;
    height: 100%;
    display: inline-block;
    vertical-align: top;
    margin-bottom: -50px;
  }
  .CodeMirror-gutter-wrapper {
    position: absolute;
    z-index: 4;
    background: none !important;
    border: none !important;
  }
  .CodeMirror-gutter-background {
    position: absolute;
    top: 0; bottom: 0;
    z-index: 4;
  }
  .CodeMirror-gutter-elt {
    position: absolute;
    cursor: default;
    z-index: 4;
  }
  .CodeMirror-gutter-wrapper ::selection { background-color: transparent }
  .CodeMirror-gutter-wrapper ::-moz-selection { background-color: transparent }
  
  .CodeMirror-lines {
    cursor: text;
    min-height: 1px; /* prevents collapsing before first draw */
  }
  .CodeMirror pre.CodeMirror-line,
  .CodeMirror pre.CodeMirror-line-like {
    /* Reset some styles that the rest of the page might have set */
    -moz-border-radius: 0; -webkit-border-radius: 0; border-radius: 0;
    border-width: 0;
    background: transparent;
    font-family: inherit;
    font-size: inherit;
    margin: 0;
    white-space: pre;
    word-wrap: normal;
    line-height: inherit;
    color: inherit;
    z-index: 2;
    position: relative;
    overflow: visible;
    -webkit-tap-highlight-color: transparent;
    -webkit-font-variant-ligatures: contextual;
    font-variant-ligatures: contextual;
  }
  .CodeMirror-wrap pre.CodeMirror-line,
  .CodeMirror-wrap pre.CodeMirror-line-like {
    word-wrap: break-word;
    white-space: pre-wrap;
    word-break: normal;
  }
  
  .CodeMirror-linebackground {
    position: absolute;
    left: 0; right: 0; top: 0; bottom: 0;
    z-index: 0;
  }
  
  .CodeMirror-linewidget {
    position: relative;
    z-index: 2;
    padding: 0.1px; /* Force widget margins to stay inside of the container */
  }
  
  .CodeMirror-widget {}
  
  .CodeMirror-rtl pre { direction: rtl; }
  
  .CodeMirror-code {
    outline: none;
  }
  
  /* Force content-box sizing for the elements where we expect it */
  .CodeMirror-scroll,
  .CodeMirror-sizer,
  .CodeMirror-gutter,
  .CodeMirror-gutters,
  .CodeMirror-linenumber {
    -moz-box-sizing: content-box;
    box-sizing: content-box;
  }
  
  .CodeMirror-measure {
    position: absolute;
    width: 100%;
    height: 0;
    overflow: hidden;
    visibility: hidden;
  }
  
  .CodeMirror-cursor {
    position: absolute;
    pointer-events: none;
  }
  .CodeMirror-measure pre { position: static; }
  
  div.CodeMirror-cursors {
    visibility: hidden;
    position: relative;
    z-index: 3;
  }
  div.CodeMirror-dragcursors {
    visibility: visible;
  }
  
  .CodeMirror-focused div.CodeMirror-cursors {
    visibility: visible;
  }
  
  .CodeMirror-selected { background: #d9d9d9; }
  .CodeMirror-focused .CodeMirror-selected { background: #d7d4f0; }
  .CodeMirror-crosshair { cursor: crosshair; }
  .CodeMirror-line::selection, .CodeMirror-line > span::selection, .CodeMirror-line > span > span::selection { background: #d7d4f0; }
  .CodeMirror-line::-moz-selection, .CodeMirror-line > span::-moz-selection, .CodeMirror-line > span > span::-moz-selection { background: #d7d4f0; }
  
  .cm-searching {
    background-color: #ffa;
    background-color: rgba(255, 255, 0, .4);
  }
  
  /* Used to force a border model for a node */
  .cm-force-border { padding-right: .1px; }
  
  @media print {
    /* Hide the cursor when printing */
    .CodeMirror div.CodeMirror-cursors {
      visibility: hidden;
    }
  }
  
  /* See issue #2901 */
  .cm-tab-wrap-hack:after { content: ''; }
  
  /* Help users use markselection to safely style text background */
  span.CodeMirror-selectedtext { background: none; }
  
  .CodeMirror-dialog {
    position: absolute;
    left: 0; right: 0;
    background: inherit;
    z-index: 15;
    padding: .1em .8em;
    overflow: hidden;
    color: inherit;
  }
  
  .CodeMirror-dialog-top {
    border-bottom: 1px solid #eee;
    top: 0;
  }
  
  .CodeMirror-dialog-bottom {
    border-top: 1px solid #eee;
    bottom: 0;
  }
  
  .CodeMirror-dialog input {
    border: none;
    outline: none;
    background: transparent;
    width: 20em;
    color: inherit;
    font-family: monospace;
  }
  
  .CodeMirror-dialog button {
    font-size: 70%;
  }
  
  .CodeMirror-foldmarker {
    color: blue;
    text-shadow: #b9f 1px 1px 2px, #b9f -1px -1px 2px, #b9f 1px -1px 2px, #b9f -1px 1px 2px;
    font-family: arial;
    line-height: .3;
    cursor: pointer;
  }
  .CodeMirror-foldgutter {
    width: .7em;
  }
  .CodeMirror-foldgutter-open,
  .CodeMirror-foldgutter-folded {
    cursor: pointer;
  }
  .CodeMirror-foldgutter-open:after {
    content: "\25BE";
  }
  .CodeMirror-foldgutter-folded:after {
    content: "\25B8";
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .CodeMirror {
    line-height: var(--jp-code-line-height);
    font-size: var(--jp-code-font-size);
    font-family: var(--jp-code-font-family);
    border: 0;
    border-radius: 0;
    height: auto;
    /* Changed to auto to autogrow */
  }
  
  .CodeMirror pre {
    padding: 0 var(--jp-code-padding);
  }
  
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-dialog {
    background-color: var(--jp-layout-color0);
    color: var(--jp-content-font-color1);
  }
  
  /* This causes https://github.com/jupyter/jupyterlab/issues/522 */
  /* May not cause it not because we changed it! */
  .CodeMirror-lines {
    padding: var(--jp-code-padding) 0;
  }
  
  .CodeMirror-linenumber {
    padding: 0 8px;
  }
  
  .jp-CodeMirrorEditor {
    cursor: text;
  }
  
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
  }
  
  /* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
  @media screen and (min-width: 2138px) and (max-width: 4319px) {
    .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
      border-left: var(--jp-code-cursor-width1) solid
        var(--jp-editor-cursor-color);
    }
  }
  
  /* When zoomed out less than 33% */
  @media screen and (min-width: 4320px) {
    .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
      border-left: var(--jp-code-cursor-width2) solid
        var(--jp-editor-cursor-color);
    }
  }
  
  .CodeMirror.jp-mod-readOnly .CodeMirror-cursor {
    display: none;
  }
  
  .CodeMirror-gutters {
    border-right: 1px solid var(--jp-border-color2);
    background-color: var(--jp-layout-color0);
  }
  
  .jp-CollaboratorCursor {
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: none;
    border-bottom: 3px solid;
    background-clip: content-box;
    margin-left: -5px;
    margin-right: -5px;
  }
  
  .CodeMirror-selectedtext.cm-searching {
    background-color: var(--jp-search-selected-match-background-color) !important;
    color: var(--jp-search-selected-match-color) !important;
  }
  
  .cm-searching {
    background-color: var(
      --jp-search-unselected-match-background-color
    ) !important;
    color: var(--jp-search-unselected-match-color) !important;
  }
  
  .CodeMirror-focused .CodeMirror-selected {
    background-color: var(--jp-editor-selected-focused-background);
  }
  
  .CodeMirror-selected {
    background-color: var(--jp-editor-selected-background);
  }
  
  .jp-CollaboratorCursor-hover {
    position: absolute;
    z-index: 1;
    transform: translateX(-50%);
    color: white;
    border-radius: 3px;
    padding-left: 4px;
    padding-right: 4px;
    padding-top: 1px;
    padding-bottom: 1px;
    text-align: center;
    font-size: var(--jp-ui-font-size1);
    white-space: nowrap;
  }
  
  .jp-CodeMirror-ruler {
    border-left: 1px dashed var(--jp-border-color2);
  }
  
  /**
   * Here is our jupyter theme for CodeMirror syntax highlighting
   * This is used in our marked.js syntax highlighting and CodeMirror itself
   * The string "jupyter" is set in ../codemirror/widget.DEFAULT_CODEMIRROR_THEME
   * This came from the classic notebook, which came form highlight.js/GitHub
   */
  
  /**
   * CodeMirror themes are handling the background/color in this way. This works
   * fine for CodeMirror editors outside the notebook, but the notebook styles
   * these things differently.
   */
  .CodeMirror.cm-s-jupyter {
    background: var(--jp-layout-color0);
    color: var(--jp-content-font-color1);
  }
  
  /* In the notebook, we want this styling to be handled by its container */
  .jp-CodeConsole .CodeMirror.cm-s-jupyter,
  .jp-Notebook .CodeMirror.cm-s-jupyter {
    background: transparent;
  }
  
  .cm-s-jupyter .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
  }
  .cm-s-jupyter span.cm-keyword {
    color: var(--jp-mirror-editor-keyword-color);
    font-weight: bold;
  }
  .cm-s-jupyter span.cm-atom {
    color: var(--jp-mirror-editor-atom-color);
  }
  .cm-s-jupyter span.cm-number {
    color: var(--jp-mirror-editor-number-color);
  }
  .cm-s-jupyter span.cm-def {
    color: var(--jp-mirror-editor-def-color);
  }
  .cm-s-jupyter span.cm-variable {
    color: var(--jp-mirror-editor-variable-color);
  }
  .cm-s-jupyter span.cm-variable-2 {
    color: var(--jp-mirror-editor-variable-2-color);
  }
  .cm-s-jupyter span.cm-variable-3 {
    color: var(--jp-mirror-editor-variable-3-color);
  }
  .cm-s-jupyter span.cm-punctuation {
    color: var(--jp-mirror-editor-punctuation-color);
  }
  .cm-s-jupyter span.cm-property {
    color: var(--jp-mirror-editor-property-color);
  }
  .cm-s-jupyter span.cm-operator {
    color: var(--jp-mirror-editor-operator-color);
    font-weight: bold;
  }
  .cm-s-jupyter span.cm-comment {
    color: var(--jp-mirror-editor-comment-color);
    font-style: italic;
  }
  .cm-s-jupyter span.cm-string {
    color: var(--jp-mirror-editor-string-color);
  }
  .cm-s-jupyter span.cm-string-2 {
    color: var(--jp-mirror-editor-string-2-color);
  }
  .cm-s-jupyter span.cm-meta {
    color: var(--jp-mirror-editor-meta-color);
  }
  .cm-s-jupyter span.cm-qualifier {
    color: var(--jp-mirror-editor-qualifier-color);
  }
  .cm-s-jupyter span.cm-builtin {
    color: var(--jp-mirror-editor-builtin-color);
  }
  .cm-s-jupyter span.cm-bracket {
    color: var(--jp-mirror-editor-bracket-color);
  }
  .cm-s-jupyter span.cm-tag {
    color: var(--jp-mirror-editor-tag-color);
  }
  .cm-s-jupyter span.cm-attribute {
    color: var(--jp-mirror-editor-attribute-color);
  }
  .cm-s-jupyter span.cm-header {
    color: var(--jp-mirror-editor-header-color);
  }
  .cm-s-jupyter span.cm-quote {
    color: var(--jp-mirror-editor-quote-color);
  }
  .cm-s-jupyter span.cm-link {
    color: var(--jp-mirror-editor-link-color);
  }
  .cm-s-jupyter span.cm-error {
    color: var(--jp-mirror-editor-error-color);
  }
  .cm-s-jupyter span.cm-hr {
    color: #999;
  }
  
  .cm-s-jupyter span.cm-tab {
    background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
    background-position: right;
    background-repeat: no-repeat;
  }
  
  .cm-s-jupyter .CodeMirror-activeline-background,
  .cm-s-jupyter .CodeMirror-gutter {
    background-color: var(--jp-layout-color2);
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | RenderedText
  |----------------------------------------------------------------------------*/
  
  :root {
    /* This is the padding value to fill the gaps between lines containing spans with background color. */
    --jp-private-code-span-padding: calc(
      (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
    );
  }
  
  .jp-RenderedText {
    text-align: left;
    padding-left: var(--jp-code-padding);
    line-height: var(--jp-code-line-height);
    font-family: var(--jp-code-font-family);
  }
  
  .jp-RenderedText pre,
  .jp-RenderedJavaScript pre,
  .jp-RenderedHTMLCommon pre {
    color: var(--jp-content-font-color1);
    font-size: var(--jp-code-font-size);
    border: none;
    margin: 0px;
    padding: 0px;
  }
  
  .jp-RenderedText pre a:link {
    text-decoration: none;
    color: var(--jp-content-link-color);
  }
  .jp-RenderedText pre a:hover {
    text-decoration: underline;
    color: var(--jp-content-link-color);
  }
  .jp-RenderedText pre a:visited {
    text-decoration: none;
    color: var(--jp-content-link-color);
  }
  
  /* console foregrounds and backgrounds */
  .jp-RenderedText pre .ansi-black-fg {
    color: #3e424d;
  }
  .jp-RenderedText pre .ansi-red-fg {
    color: #e75c58;
  }
  .jp-RenderedText pre .ansi-green-fg {
    color: #00a250;
  }
  .jp-RenderedText pre .ansi-yellow-fg {
    color: #ddb62b;
  }
  .jp-RenderedText pre .ansi-blue-fg {
    color: #208ffb;
  }
  .jp-RenderedText pre .ansi-magenta-fg {
    color: #d160c4;
  }
  .jp-RenderedText pre .ansi-cyan-fg {
    color: #60c6c8;
  }
  .jp-RenderedText pre .ansi-white-fg {
    color: #c5c1b4;
  }
  
  .jp-RenderedText pre .ansi-black-bg {
    background-color: #3e424d;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-red-bg {
    background-color: #e75c58;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-green-bg {
    background-color: #00a250;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-yellow-bg {
    background-color: #ddb62b;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-blue-bg {
    background-color: #208ffb;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-magenta-bg {
    background-color: #d160c4;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-cyan-bg {
    background-color: #60c6c8;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-white-bg {
    background-color: #c5c1b4;
    padding: var(--jp-private-code-span-padding) 0;
  }
  
  .jp-RenderedText pre .ansi-black-intense-fg {
    color: #282c36;
  }
  .jp-RenderedText pre .ansi-red-intense-fg {
    color: #b22b31;
  }
  .jp-RenderedText pre .ansi-green-intense-fg {
    color: #007427;
  }
  .jp-RenderedText pre .ansi-yellow-intense-fg {
    color: #b27d12;
  }
  .jp-RenderedText pre .ansi-blue-intense-fg {
    color: #0065ca;
  }
  .jp-RenderedText pre .ansi-magenta-intense-fg {
    color: #a03196;
  }
  .jp-RenderedText pre .ansi-cyan-intense-fg {
    color: #258f8f;
  }
  .jp-RenderedText pre .ansi-white-intense-fg {
    color: #a1a6b2;
  }
  
  .jp-RenderedText pre .ansi-black-intense-bg {
    background-color: #282c36;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-red-intense-bg {
    background-color: #b22b31;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-green-intense-bg {
    background-color: #007427;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-yellow-intense-bg {
    background-color: #b27d12;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-blue-intense-bg {
    background-color: #0065ca;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-magenta-intense-bg {
    background-color: #a03196;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-cyan-intense-bg {
    background-color: #258f8f;
    padding: var(--jp-private-code-span-padding) 0;
  }
  .jp-RenderedText pre .ansi-white-intense-bg {
    background-color: #a1a6b2;
    padding: var(--jp-private-code-span-padding) 0;
  }
  
  .jp-RenderedText pre .ansi-default-inverse-fg {
    color: var(--jp-ui-inverse-font-color0);
  }
  .jp-RenderedText pre .ansi-default-inverse-bg {
    background-color: var(--jp-inverse-layout-color0);
    padding: var(--jp-private-code-span-padding) 0;
  }
  
  .jp-RenderedText pre .ansi-bold {
    font-weight: bold;
  }
  .jp-RenderedText pre .ansi-underline {
    text-decoration: underline;
  }
  
  .jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
    background: var(--jp-rendermime-error-background);
    padding-top: var(--jp-code-padding);
  }
  
  /*-----------------------------------------------------------------------------
  | RenderedLatex
  |----------------------------------------------------------------------------*/
  
  .jp-RenderedLatex {
    color: var(--jp-content-font-color1);
    font-size: var(--jp-content-font-size1);
    line-height: var(--jp-content-line-height);
  }
  
  /* Left-justify outputs.*/
  .jp-OutputArea-output.jp-RenderedLatex {
    padding: var(--jp-code-padding);
    text-align: left;
  }
  
  /*-----------------------------------------------------------------------------
  | RenderedHTML
  |----------------------------------------------------------------------------*/
  
  .jp-RenderedHTMLCommon {
    color: var(--jp-content-font-color1);
    font-family: var(--jp-content-font-family);
    font-size: var(--jp-content-font-size1);
    line-height: var(--jp-content-line-height);
    /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
    padding-right: 20px;
  }
  
  .jp-RenderedHTMLCommon em {
    font-style: italic;
  }
  
  .jp-RenderedHTMLCommon strong {
    font-weight: bold;
  }
  
  .jp-RenderedHTMLCommon u {
    text-decoration: underline;
  }
  
  .jp-RenderedHTMLCommon a:link {
    text-decoration: none;
    color: var(--jp-content-link-color);
  }
  
  .jp-RenderedHTMLCommon a:hover {
    text-decoration: underline;
    color: var(--jp-content-link-color);
  }
  
  .jp-RenderedHTMLCommon a:visited {
    text-decoration: none;
    color: var(--jp-content-link-color);
  }
  
  /* Headings */
  
  .jp-RenderedHTMLCommon h1,
  .jp-RenderedHTMLCommon h2,
  .jp-RenderedHTMLCommon h3,
  .jp-RenderedHTMLCommon h4,
  .jp-RenderedHTMLCommon h5,
  .jp-RenderedHTMLCommon h6 {
    line-height: var(--jp-content-heading-line-height);
    font-weight: var(--jp-content-heading-font-weight);
    font-style: normal;
    margin: var(--jp-content-heading-margin-top) 0
      var(--jp-content-heading-margin-bottom) 0;
  }
  
  .jp-RenderedHTMLCommon h1:first-child,
  .jp-RenderedHTMLCommon h2:first-child,
  .jp-RenderedHTMLCommon h3:first-child,
  .jp-RenderedHTMLCommon h4:first-child,
  .jp-RenderedHTMLCommon h5:first-child,
  .jp-RenderedHTMLCommon h6:first-child {
    margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
  }
  
  .jp-RenderedHTMLCommon h1:last-child,
  .jp-RenderedHTMLCommon h2:last-child,
  .jp-RenderedHTMLCommon h3:last-child,
  .jp-RenderedHTMLCommon h4:last-child,
  .jp-RenderedHTMLCommon h5:last-child,
  .jp-RenderedHTMLCommon h6:last-child {
    margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
  }
  
  .jp-RenderedHTMLCommon h1 {
    font-size: var(--jp-content-font-size5);
  }
  
  .jp-RenderedHTMLCommon h2 {
    font-size: var(--jp-content-font-size4);
  }
  
  .jp-RenderedHTMLCommon h3 {
    font-size: var(--jp-content-font-size3);
  }
  
  .jp-RenderedHTMLCommon h4 {
    font-size: var(--jp-content-font-size2);
  }
  
  .jp-RenderedHTMLCommon h5 {
    font-size: var(--jp-content-font-size1);
  }
  
  .jp-RenderedHTMLCommon h6 {
    font-size: var(--jp-content-font-size0);
  }
  
  /* Lists */
  
  .jp-RenderedHTMLCommon ul:not(.list-inline),
  .jp-RenderedHTMLCommon ol:not(.list-inline) {
    padding-left: 2em;
  }
  
  .jp-RenderedHTMLCommon ul {
    list-style: disc;
  }
  
  .jp-RenderedHTMLCommon ul ul {
    list-style: square;
  }
  
  .jp-RenderedHTMLCommon ul ul ul {
    list-style: circle;
  }
  
  .jp-RenderedHTMLCommon ol {
    list-style: decimal;
  }
  
  .jp-RenderedHTMLCommon ol ol {
    list-style: upper-alpha;
  }
  
  .jp-RenderedHTMLCommon ol ol ol {
    list-style: lower-alpha;
  }
  
  .jp-RenderedHTMLCommon ol ol ol ol {
    list-style: lower-roman;
  }
  
  .jp-RenderedHTMLCommon ol ol ol ol ol {
    list-style: decimal;
  }
  
  .jp-RenderedHTMLCommon ol,
  .jp-RenderedHTMLCommon ul {
    margin-bottom: 1em;
  }
  
  .jp-RenderedHTMLCommon ul ul,
  .jp-RenderedHTMLCommon ul ol,
  .jp-RenderedHTMLCommon ol ul,
  .jp-RenderedHTMLCommon ol ol {
    margin-bottom: 0em;
  }
  
  .jp-RenderedHTMLCommon hr {
    color: var(--jp-border-color2);
    background-color: var(--jp-border-color1);
    margin-top: 1em;
    margin-bottom: 1em;
  }
  
  .jp-RenderedHTMLCommon > pre {
    margin: 1.5em 2em;
  }
  
  .jp-RenderedHTMLCommon pre,
  .jp-RenderedHTMLCommon code {
    border: 0;
    background-color: var(--jp-layout-color0);
    color: var(--jp-content-font-color1);
    font-family: var(--jp-code-font-family);
    font-size: inherit;
    line-height: var(--jp-code-line-height);
    padding: 0;
    white-space: pre-wrap;
  }
  
  .jp-RenderedHTMLCommon :not(pre) > code {
    background-color: var(--jp-layout-color2);
    padding: 1px 5px;
  }
  
  /* Tables */
  
  .jp-RenderedHTMLCommon table {
    border-collapse: collapse;
    border-spacing: 0;
    border: none;
    color: var(--jp-ui-font-color1);
    font-size: 12px;
    table-layout: fixed;
    margin-left: auto;
    margin-right: auto;
  }
  
  .jp-RenderedHTMLCommon thead {
    border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
    vertical-align: bottom;
  }
  
  .jp-RenderedHTMLCommon td,
  .jp-RenderedHTMLCommon th,
  .jp-RenderedHTMLCommon tr {
    vertical-align: middle;
    padding: 0.5em 0.5em;
    line-height: normal;
    white-space: normal;
    max-width: none;
    border: none;
  }
  
  .jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
  .jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
    max-width: none;
  }
  
  :not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
  :not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
  :not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
    text-align: right;
  }
  
  .jp-RenderedHTMLCommon th {
    font-weight: bold;
  }
  
  .jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
    background: var(--jp-layout-color0);
  }
  
  .jp-RenderedHTMLCommon tbody tr:nth-child(even) {
    background: var(--jp-rendermime-table-row-background);
  }
  
  .jp-RenderedHTMLCommon tbody tr:hover {
    background: var(--jp-rendermime-table-row-hover-background);
  }
  
  .jp-RenderedHTMLCommon table {
    margin-bottom: 1em;
  }
  
  .jp-RenderedHTMLCommon p {
    text-align: left;
    margin: 0px;
  }
  
  .jp-RenderedHTMLCommon p {
    margin-bottom: 1em;
  }
  
  .jp-RenderedHTMLCommon img {
    -moz-force-broken-image-icon: 1;
  }
  
  /* Restrict to direct children as other images could be nested in other content. */
  .jp-RenderedHTMLCommon > img {
    display: block;
    margin-left: 0;
    margin-right: 0;
    margin-bottom: 1em;
  }
  
  /* Change color behind transparent images if they need it... */
  [data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
    background-color: var(--jp-inverse-layout-color1);
  }
  [data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
    background-color: var(--jp-inverse-layout-color1);
  }
  /* ...or leave it untouched if they don't */
  [data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-dark-background {
  }
  [data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
  }
  
  .jp-RenderedHTMLCommon img,
  .jp-RenderedImage img,
  .jp-RenderedHTMLCommon svg,
  .jp-RenderedSVG svg {
    max-width: 100%;
    height: auto;
  }
  
  .jp-RenderedHTMLCommon img.jp-mod-unconfined,
  .jp-RenderedImage img.jp-mod-unconfined,
  .jp-RenderedHTMLCommon svg.jp-mod-unconfined,
  .jp-RenderedSVG svg.jp-mod-unconfined {
    max-width: none;
  }
  
  .jp-RenderedHTMLCommon .alert {
    padding: var(--jp-notebook-padding);
    border: var(--jp-border-width) solid transparent;
    border-radius: var(--jp-border-radius);
    margin-bottom: 1em;
  }
  
  .jp-RenderedHTMLCommon .alert-info {
    color: var(--jp-info-color0);
    background-color: var(--jp-info-color3);
    border-color: var(--jp-info-color2);
  }
  .jp-RenderedHTMLCommon .alert-info hr {
    border-color: var(--jp-info-color3);
  }
  .jp-RenderedHTMLCommon .alert-info > p:last-child,
  .jp-RenderedHTMLCommon .alert-info > ul:last-child {
    margin-bottom: 0;
  }
  
  .jp-RenderedHTMLCommon .alert-warning {
    color: var(--jp-warn-color0);
    background-color: var(--jp-warn-color3);
    border-color: var(--jp-warn-color2);
  }
  .jp-RenderedHTMLCommon .alert-warning hr {
    border-color: var(--jp-warn-color3);
  }
  .jp-RenderedHTMLCommon .alert-warning > p:last-child,
  .jp-RenderedHTMLCommon .alert-warning > ul:last-child {
    margin-bottom: 0;
  }
  
  .jp-RenderedHTMLCommon .alert-success {
    color: var(--jp-success-color0);
    background-color: var(--jp-success-color3);
    border-color: var(--jp-success-color2);
  }
  .jp-RenderedHTMLCommon .alert-success hr {
    border-color: var(--jp-success-color3);
  }
  .jp-RenderedHTMLCommon .alert-success > p:last-child,
  .jp-RenderedHTMLCommon .alert-success > ul:last-child {
    margin-bottom: 0;
  }
  
  .jp-RenderedHTMLCommon .alert-danger {
    color: var(--jp-error-color0);
    background-color: var(--jp-error-color3);
    border-color: var(--jp-error-color2);
  }
  .jp-RenderedHTMLCommon .alert-danger hr {
    border-color: var(--jp-error-color3);
  }
  .jp-RenderedHTMLCommon .alert-danger > p:last-child,
  .jp-RenderedHTMLCommon .alert-danger > ul:last-child {
    margin-bottom: 0;
  }
  
  .jp-RenderedHTMLCommon blockquote {
    margin: 1em 2em;
    padding: 0 1em;
    border-left: 5px solid var(--jp-border-color2);
  }
  
  a.jp-InternalAnchorLink {
    visibility: hidden;
    margin-left: 8px;
    color: var(--md-blue-800);
  }
  
  h1:hover .jp-InternalAnchorLink,
  h2:hover .jp-InternalAnchorLink,
  h3:hover .jp-InternalAnchorLink,
  h4:hover .jp-InternalAnchorLink,
  h5:hover .jp-InternalAnchorLink,
  h6:hover .jp-InternalAnchorLink {
    visibility: visible;
  }
  
  .jp-RenderedHTMLCommon kbd {
    background-color: var(--jp-rendermime-table-row-background);
    border: 1px solid var(--jp-border-color0);
    border-bottom-color: var(--jp-border-color2);
    border-radius: 3px;
    box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
    display: inline-block;
    font-size: 0.8em;
    line-height: 1em;
    padding: 0.2em 0.5em;
  }
  
  /* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
   * At the bottom of cells this is a bit too much as there is also spacing
   * between cells. Going all the way to 0 gets too tight between markdown and
   * code cells.
   */
  .jp-RenderedHTMLCommon > *:last-child {
    margin-bottom: 0.5em;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-MimeDocument {
    outline: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Variables
  |----------------------------------------------------------------------------*/
  
  :root {
    --jp-private-filebrowser-button-height: 28px;
    --jp-private-filebrowser-button-width: 48px;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-FileBrowser {
    display: flex;
    flex-direction: column;
    color: var(--jp-ui-font-color1);
    background: var(--jp-layout-color1);
    /* This is needed so that all font sizing of children done in ems is
     * relative to this base size */
    font-size: var(--jp-ui-font-size1);
  }
  
  .jp-FileBrowser-toolbar.jp-Toolbar {
    border-bottom: none;
    height: auto;
    margin: var(--jp-toolbar-header-margin);
    box-shadow: none;
  }
  
  .jp-BreadCrumbs {
    flex: 0 0 auto;
    margin: 8px 12px 8px 12px;
  }
  
  .jp-BreadCrumbs-item {
    margin: 0px 2px;
    padding: 0px 2px;
    border-radius: var(--jp-border-radius);
    cursor: pointer;
  }
  
  .jp-BreadCrumbs-item:hover {
    background-color: var(--jp-layout-color2);
  }
  
  .jp-BreadCrumbs-item:first-child {
    margin-left: 0px;
  }
  
  .jp-BreadCrumbs-item.jp-mod-dropTarget {
    background-color: var(--jp-brand-color2);
    opacity: 0.7;
  }
  
  /*-----------------------------------------------------------------------------
  | Buttons
  |----------------------------------------------------------------------------*/
  
  .jp-FileBrowser-toolbar.jp-Toolbar {
    padding: 0px;
    margin: 8px 12px 0px 12px;
  }
  
  .jp-FileBrowser-toolbar.jp-Toolbar {
    justify-content: flex-start;
  }
  
  .jp-FileBrowser-toolbar.jp-Toolbar .jp-Toolbar-item {
    flex: 0 0 auto;
    padding-left: 0px;
    padding-right: 2px;
  }
  
  .jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
    width: 40px;
  }
  
  .jp-FileBrowser-toolbar.jp-Toolbar
    .jp-Toolbar-item:first-child
    .jp-ToolbarButtonComponent {
    width: 72px;
    background: var(--jp-brand-color1);
  }
  
  .jp-FileBrowser-toolbar.jp-Toolbar
    .jp-Toolbar-item:first-child
    .jp-ToolbarButtonComponent
    .jp-icon3 {
    fill: white;
  }
  
  /*-----------------------------------------------------------------------------
  | Other styles
  |----------------------------------------------------------------------------*/
  
  .jp-FileDialog.jp-mod-conflict input {
    color: red;
  }
  
  .jp-FileDialog .jp-new-name-title {
    margin-top: 12px;
  }
  
  .jp-LastModified-hidden {
    display: none;
  }
  
  .jp-FileBrowser-filterBox {
    padding: 0px;
    flex: 0 0 auto;
    margin: 8px 12px 0px 12px;
  }
  
  /*-----------------------------------------------------------------------------
  | DirListing
  |----------------------------------------------------------------------------*/
  
  .jp-DirListing {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    outline: 0;
  }
  
  .jp-DirListing-header {
    flex: 0 0 auto;
    display: flex;
    flex-direction: row;
    overflow: hidden;
    border-top: var(--jp-border-width) solid var(--jp-border-color2);
    border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
    box-shadow: var(--jp-toolbar-box-shadow);
    z-index: 2;
  }
  
  .jp-DirListing-headerItem {
    padding: 4px 12px 2px 12px;
    font-weight: 500;
  }
  
  .jp-DirListing-headerItem:hover {
    background: var(--jp-layout-color2);
  }
  
  .jp-DirListing-headerItem.jp-id-name {
    flex: 1 0 84px;
  }
  
  .jp-DirListing-headerItem.jp-id-modified {
    flex: 0 0 112px;
    border-left: var(--jp-border-width) solid var(--jp-border-color2);
    text-align: right;
  }
  
  .jp-id-narrow {
    display: none;
    flex: 0 0 5px;
    padding: 4px 4px;
    border-left: var(--jp-border-width) solid var(--jp-border-color2);
    text-align: right;
    color: var(--jp-border-color2);
  }
  
  .jp-DirListing-narrow .jp-id-narrow {
    display: block;
  }
  
  .jp-DirListing-narrow .jp-id-modified,
  .jp-DirListing-narrow .jp-DirListing-itemModified {
    display: none;
  }
  
  .jp-DirListing-headerItem.jp-mod-selected {
    font-weight: 600;
  }
  
  /* increase specificity to override bundled default */
  .jp-DirListing-content {
    flex: 1 1 auto;
    margin: 0;
    padding: 0;
    list-style-type: none;
    overflow: auto;
    background-color: var(--jp-layout-color1);
  }
  
  .jp-DirListing-content mark {
    color: var(--jp-ui-font-color0);
    background-color: transparent;
    font-weight: bold;
  }
  
  /* Style the directory listing content when a user drops a file to upload */
  .jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
    outline: 5px dashed rgba(128, 128, 128, 0.5);
    outline-offset: -10px;
    cursor: copy;
  }
  
  .jp-DirListing-item {
    display: flex;
    flex-direction: row;
    padding: 4px 12px;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  .jp-DirListing-item[data-is-dot] {
    opacity: 75%;
  }
  
  .jp-DirListing-item.jp-mod-selected {
    color: white;
    background: var(--jp-brand-color1);
  }
  
  .jp-DirListing-item.jp-mod-dropTarget {
    background: var(--jp-brand-color3);
  }
  
  .jp-DirListing-item:hover:not(.jp-mod-selected) {
    background: var(--jp-layout-color2);
  }
  
  .jp-DirListing-itemIcon {
    flex: 0 0 20px;
    margin-right: 4px;
  }
  
  .jp-DirListing-itemText {
    flex: 1 0 64px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    user-select: none;
  }
  
  .jp-DirListing-itemModified {
    flex: 0 0 125px;
    text-align: right;
  }
  
  .jp-DirListing-editor {
    flex: 1 0 64px;
    outline: none;
    border: none;
  }
  
  .jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon:before {
    color: limegreen;
    content: '\25CF';
    font-size: 8px;
    position: absolute;
    left: -8px;
  }
  
  .jp-DirListing-item.lm-mod-drag-image,
  .jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
    font-size: var(--jp-ui-font-size1);
    padding-left: 4px;
    margin-left: 4px;
    width: 160px;
    background-color: var(--jp-ui-inverse-font-color2);
    box-shadow: var(--jp-elevation-z2);
    border-radius: 0px;
    color: var(--jp-ui-font-color1);
    transform: translateX(-40%) translateY(-58%);
  }
  
  .jp-DirListing-deadSpace {
    flex: 1 1 auto;
    margin: 0;
    padding: 0;
    list-style-type: none;
    overflow: auto;
    background-color: var(--jp-layout-color1);
  }
  
  .jp-Document {
    min-width: 120px;
    min-height: 120px;
    outline: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Private CSS variables
  |----------------------------------------------------------------------------*/
  
  :root {
  }
  
  /*-----------------------------------------------------------------------------
  | Main OutputArea
  | OutputArea has a list of Outputs
  |----------------------------------------------------------------------------*/
  
  .jp-OutputArea {
    overflow-y: auto;
  }
  
  .jp-OutputArea-child {
    display: flex;
    flex-direction: row;
  }
  
  .jp-OutputPrompt {
    flex: 0 0 var(--jp-cell-prompt-width);
    color: var(--jp-cell-outprompt-font-color);
    font-family: var(--jp-cell-prompt-font-family);
    padding: var(--jp-code-padding);
    letter-spacing: var(--jp-cell-prompt-letter-spacing);
    line-height: var(--jp-code-line-height);
    font-size: var(--jp-code-font-size);
    border: var(--jp-border-width) solid transparent;
    opacity: var(--jp-cell-prompt-opacity);
    /* Right align prompt text, don't wrap to handle large prompt numbers */
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    /* Disable text selection */
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  .jp-OutputArea-output {
    height: auto;
    overflow: auto;
    user-select: text;
    -moz-user-select: text;
    -webkit-user-select: text;
    -ms-user-select: text;
  }
  
  .jp-OutputArea-child .jp-OutputArea-output {
    flex-grow: 1;
    flex-shrink: 1;
  }
  
  /**
   * Isolated output.
   */
  .jp-OutputArea-output.jp-mod-isolated {
    width: 100%;
    display: block;
  }
  
  /*
  When drag events occur, `p-mod-override-cursor` is added to the body.
  Because iframes steal all cursor events, the following two rules are necessary
  to suppress pointer events while resize drags are occurring. There may be a
  better solution to this problem.
  */
  body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
    position: relative;
  }
  
  body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: transparent;
  }
  
  /* pre */
  
  .jp-OutputArea-output pre {
    border: none;
    margin: 0px;
    padding: 0px;
    overflow-x: auto;
    overflow-y: auto;
    word-break: break-all;
    word-wrap: break-word;
    white-space: pre-wrap;
  }
  
  /* tables */
  
  .jp-OutputArea-output.jp-RenderedHTMLCommon table {
    margin-left: 0;
    margin-right: 0;
  }
  
  /* description lists */
  
  .jp-OutputArea-output dl,
  .jp-OutputArea-output dt,
  .jp-OutputArea-output dd {
    display: block;
  }
  
  .jp-OutputArea-output dl {
    width: 100%;
    overflow: hidden;
    padding: 0;
    margin: 0;
  }
  
  .jp-OutputArea-output dt {
    font-weight: bold;
    float: left;
    width: 20%;
    padding: 0;
    margin: 0;
  }
  
  .jp-OutputArea-output dd {
    float: left;
    width: 80%;
    padding: 0;
    margin: 0;
  }
  
  /* Hide the gutter in case of
   *  - nested output areas (e.g. in the case of output widgets)
   *  - mirrored output areas
   */
  .jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
    display: none;
  }
  
  /*-----------------------------------------------------------------------------
  | executeResult is added to any Output-result for the display of the object
  | returned by a cell
  |----------------------------------------------------------------------------*/
  
  .jp-OutputArea-output.jp-OutputArea-executeResult {
    margin-left: 0px;
    flex: 1 1 auto;
  }
  
  /* Text output with the Out[] prompt needs a top padding to match the
   * alignment of the Out[] prompt itself.
   */
  .jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
    padding-top: var(--jp-code-padding);
    border-top: var(--jp-border-width) solid transparent;
  }
  
  /*-----------------------------------------------------------------------------
  | The Stdin output
  |----------------------------------------------------------------------------*/
  
  .jp-OutputArea-stdin {
    line-height: var(--jp-code-line-height);
    padding-top: var(--jp-code-padding);
    display: flex;
  }
  
  .jp-Stdin-prompt {
    color: var(--jp-content-font-color0);
    padding-right: var(--jp-code-padding);
    vertical-align: baseline;
    flex: 0 0 auto;
  }
  
  .jp-Stdin-input {
    font-family: var(--jp-code-font-family);
    font-size: inherit;
    color: inherit;
    background-color: inherit;
    width: 42%;
    min-width: 200px;
    /* make sure input baseline aligns with prompt */
    vertical-align: baseline;
    /* padding + margin = 0.5em between prompt and cursor */
    padding: 0em 0.25em;
    margin: 0em 0.25em;
    flex: 0 0 70%;
  }
  
  .jp-Stdin-input:focus {
    box-shadow: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Output Area View
  |----------------------------------------------------------------------------*/
  
  .jp-LinkedOutputView .jp-OutputArea {
    height: 100%;
    display: block;
  }
  
  .jp-LinkedOutputView .jp-OutputArea-output:only-child {
    height: 100%;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  .jp-Collapser {
    flex: 0 0 var(--jp-cell-collapser-width);
    padding: 0px;
    margin: 0px;
    border: none;
    outline: none;
    background: transparent;
    border-radius: var(--jp-border-radius);
    opacity: 1;
  }
  
  .jp-Collapser-child {
    display: block;
    width: 100%;
    box-sizing: border-box;
    /* height: 100% doesn't work because the height of its parent is computed from content */
    position: absolute;
    top: 0px;
    bottom: 0px;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Header/Footer
  |----------------------------------------------------------------------------*/
  
  /* Hidden by zero height by default */
  .jp-CellHeader,
  .jp-CellFooter {
    height: 0px;
    width: 100%;
    padding: 0px;
    margin: 0px;
    border: none;
    outline: none;
    background: transparent;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Input
  |----------------------------------------------------------------------------*/
  
  /* All input areas */
  .jp-InputArea {
    display: flex;
    flex-direction: row;
    overflow: hidden;
  }
  
  .jp-InputArea-editor {
    flex: 1 1 auto;
    overflow: hidden;
  }
  
  .jp-InputArea-editor {
    /* This is the non-active, default styling */
    border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
    border-radius: 0px;
    background: var(--jp-cell-editor-background);
  }
  
  .jp-InputPrompt {
    flex: 0 0 var(--jp-cell-prompt-width);
    color: var(--jp-cell-inprompt-font-color);
    font-family: var(--jp-cell-prompt-font-family);
    padding: var(--jp-code-padding);
    letter-spacing: var(--jp-cell-prompt-letter-spacing);
    opacity: var(--jp-cell-prompt-opacity);
    line-height: var(--jp-code-line-height);
    font-size: var(--jp-code-font-size);
    border: var(--jp-border-width) solid transparent;
    opacity: var(--jp-cell-prompt-opacity);
    /* Right align prompt text, don't wrap to handle large prompt numbers */
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    /* Disable text selection */
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Placeholder
  |----------------------------------------------------------------------------*/
  
  .jp-Placeholder {
    display: flex;
    flex-direction: row;
    flex: 1 1 auto;
  }
  
  .jp-Placeholder-prompt {
    box-sizing: border-box;
  }
  
  .jp-Placeholder-content {
    flex: 1 1 auto;
    border: none;
    background: transparent;
    height: 20px;
    box-sizing: border-box;
  }
  
  .jp-Placeholder-content .jp-MoreHorizIcon {
    width: 32px;
    height: 16px;
    border: 1px solid transparent;
    border-radius: var(--jp-border-radius);
  }
  
  .jp-Placeholder-content .jp-MoreHorizIcon:hover {
    border: 1px solid var(--jp-border-color1);
    box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.25);
    background-color: var(--jp-layout-color0);
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Private CSS variables
  |----------------------------------------------------------------------------*/
  
  :root {
    --jp-private-cell-scrolling-output-offset: 5px;
  }
  
  /*-----------------------------------------------------------------------------
  | Cell
  |----------------------------------------------------------------------------*/
  
  .jp-Cell {
    padding: var(--jp-cell-padding);
    margin: 0px;
    border: none;
    outline: none;
    background: transparent;
  }
  
  /*-----------------------------------------------------------------------------
  | Common input/output
  |----------------------------------------------------------------------------*/
  
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: flex;
    flex-direction: row;
    padding: 0px;
    margin: 0px;
    /* Added to reveal the box-shadow on the input and output collapsers. */
    overflow: visible;
  }
  
  /* Only input/output areas inside cells */
  .jp-Cell-inputArea,
  .jp-Cell-outputArea {
    flex: 1 1 auto;
  }
  
  /*-----------------------------------------------------------------------------
  | Collapser
  |----------------------------------------------------------------------------*/
  
  /* Make the output collapser disappear when there is not output, but do so
   * in a manner that leaves it in the layout and preserves its width.
   */
  .jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
    border: none !important;
    background: transparent !important;
  }
  
  .jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
    min-height: var(--jp-cell-collapser-min-height);
  }
  
  /*-----------------------------------------------------------------------------
  | Output
  |----------------------------------------------------------------------------*/
  
  /* Put a space between input and output when there IS output */
  .jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
    margin-top: 5px;
  }
  
  .jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
    overflow-y: auto;
    max-height: 200px;
    box-shadow: inset 0 0 6px 2px rgba(0, 0, 0, 0.3);
    margin-left: var(--jp-private-cell-scrolling-output-offset);
  }
  
  .jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
    flex: 0 0
      calc(
        var(--jp-cell-prompt-width) -
          var(--jp-private-cell-scrolling-output-offset)
      );
  }
  
  /*-----------------------------------------------------------------------------
  | CodeCell
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | MarkdownCell
  |----------------------------------------------------------------------------*/
  
  .jp-MarkdownOutput {
    flex: 1 1 auto;
    margin-top: 0;
    margin-bottom: 0;
    padding-left: var(--jp-code-padding);
  }
  
  .jp-MarkdownOutput.jp-RenderedHTMLCommon {
    overflow: auto;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Variables
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  
  /*-----------------------------------------------------------------------------
  | Styles
  |----------------------------------------------------------------------------*/
  
  .jp-NotebookPanel-toolbar {
    padding: 2px;
  }
  
  .jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
    border: none;
    box-shadow: none;
  }
  
  .jp-Notebook-toolbarCellTypeDropdown select {
    height: 24px;
    font-size: var(--jp-ui-font-size1);
    line-height: 14px;
    border-radius: 0;
    display: block;
  }
  
  .jp-Notebook-toolbarCellTypeDropdown span {
    top: 5px !important;
  }
  
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Private CSS variables
  |----------------------------------------------------------------------------*/
  
  :root {
    --jp-private-notebook-dragImage-width: 304px;
    --jp-private-notebook-dragImage-height: 36px;
    --jp-private-notebook-selected-color: var(--md-blue-400);
    --jp-private-notebook-active-color: var(--md-green-400);
  }
  
  /*-----------------------------------------------------------------------------
  | Imports
  |----------------------------------------------------------------------------*/
  
  /*-----------------------------------------------------------------------------
  | Notebook
  |----------------------------------------------------------------------------*/
  
  .jp-NotebookPanel {
    display: block;
    height: 100%;
  }
  
  .jp-NotebookPanel.jp-Document {
    min-width: 240px;
    min-height: 120px;
  }
  
  .jp-Notebook {
    padding: var(--jp-notebook-padding);
    outline: none;
    overflow: auto;
    background: var(--jp-layout-color0);
  }
  
  .jp-Notebook.jp-mod-scrollPastEnd::after {
    display: block;
    content: '';
    min-height: var(--jp-notebook-scroll-padding);
  }
  
  .jp-Notebook .jp-Cell {
    overflow: visible;
  }
  
  .jp-Notebook .jp-Cell .jp-InputPrompt {
    cursor: move;
  }
  
  /*-----------------------------------------------------------------------------
  | Notebook state related styling
  |
  | The notebook and cells each have states, here are the possibilities:
  |
  | - Notebook
  |   - Command
  |   - Edit
  | - Cell
  |   - None
  |   - Active (only one can be active)
  |   - Selected (the cells actions are applied to)
  |   - Multiselected (when multiple selected, the cursor)
  |   - No outputs
  |----------------------------------------------------------------------------*/
  
  /* Command or edit modes */
  
  .jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
    opacity: var(--jp-cell-prompt-not-active-opacity);
    color: var(--jp-cell-prompt-not-active-font-color);
  }
  
  .jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
    opacity: var(--jp-cell-prompt-not-active-opacity);
    color: var(--jp-cell-prompt-not-active-font-color);
  }
  
  /* cell is active */
  .jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
    background: var(--jp-brand-color1);
  }
  
  /* collapser is hovered */
  .jp-Notebook .jp-Cell .jp-Collapser:hover {
    box-shadow: var(--jp-elevation-z2);
    background: var(--jp-brand-color1);
    opacity: var(--jp-cell-collapser-not-active-hover-opacity);
  }
  
  /* cell is active and collapser is hovered */
  .jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
    background: var(--jp-brand-color0);
    opacity: 1;
  }
  
  /* Command mode */
  
  .jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
    background: var(--jp-notebook-multiselected-color);
  }
  
  .jp-Notebook.jp-mod-commandMode
    .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
    background: transparent;
  }
  
  /* Edit mode */
  
  .jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
    border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
    box-shadow: var(--jp-input-box-shadow);
    background-color: var(--jp-cell-editor-active-background);
  }
  
  /*-----------------------------------------------------------------------------
  | Notebook drag and drop
  |----------------------------------------------------------------------------*/
  
  .jp-Notebook-cell.jp-mod-dropSource {
    opacity: 0.5;
  }
  
  .jp-Notebook-cell.jp-mod-dropTarget,
  .jp-Notebook.jp-mod-commandMode
    .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
    border-top-color: var(--jp-private-notebook-selected-color);
    border-top-style: solid;
    border-top-width: 2px;
  }
  
  .jp-dragImage {
    display: flex;
    flex-direction: row;
    width: var(--jp-private-notebook-dragImage-width);
    height: var(--jp-private-notebook-dragImage-height);
    border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
    background: var(--jp-cell-editor-background);
    overflow: visible;
  }
  
  .jp-dragImage-singlePrompt {
    box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
  }
  
  .jp-dragImage .jp-dragImage-content {
    flex: 1 1 auto;
    z-index: 2;
    font-size: var(--jp-code-font-size);
    font-family: var(--jp-code-font-family);
    line-height: var(--jp-code-line-height);
    padding: var(--jp-code-padding);
    border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
    background: var(--jp-cell-editor-background-color);
    color: var(--jp-content-font-color3);
    text-align: left;
    margin: 4px 4px 4px 0px;
  }
  
  .jp-dragImage .jp-dragImage-prompt {
    flex: 0 0 auto;
    min-width: 36px;
    color: var(--jp-cell-inprompt-font-color);
    padding: var(--jp-code-padding);
    padding-left: 12px;
    font-family: var(--jp-cell-prompt-font-family);
    letter-spacing: var(--jp-cell-prompt-letter-spacing);
    line-height: 1.9;
    font-size: var(--jp-code-font-size);
    border: var(--jp-border-width) solid transparent;
  }
  
  .jp-dragImage-multipleBack {
    z-index: -1;
    position: absolute;
    height: 32px;
    width: 300px;
    top: 8px;
    left: 8px;
    background: var(--jp-layout-color2);
    border: var(--jp-border-width) solid var(--jp-input-border-color);
    box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
  }
  
  /*-----------------------------------------------------------------------------
  | Cell toolbar
  |----------------------------------------------------------------------------*/
  
  .jp-NotebookTools {
    display: block;
    min-width: var(--jp-sidebar-min-width);
    color: var(--jp-ui-font-color1);
    background: var(--jp-layout-color1);
    /* This is needed so that all font sizing of children done in ems is
      * relative to this base size */
    font-size: var(--jp-ui-font-size1);
    overflow: auto;
  }
  
  .jp-NotebookTools-tool {
    padding: 0px 12px 0 12px;
  }
  
  .jp-ActiveCellTool {
    padding: 12px;
    background-color: var(--jp-layout-color1);
    border-top: none !important;
  }
  
  .jp-ActiveCellTool .jp-InputArea-prompt {
    flex: 0 0 auto;
    padding-left: 0px;
  }
  
  .jp-ActiveCellTool .jp-InputArea-editor {
    flex: 1 1 auto;
    background: var(--jp-cell-editor-background);
    border-color: var(--jp-cell-editor-border-color);
  }
  
  .jp-ActiveCellTool .jp-InputArea-editor .CodeMirror {
    background: transparent;
  }
  
  .jp-MetadataEditorTool {
    flex-direction: column;
    padding: 12px 0px 12px 0px;
  }
  
  .jp-RankedPanel > :not(:first-child) {
    margin-top: 12px;
  }
  
  .jp-KeySelector select.jp-mod-styled {
    font-size: var(--jp-ui-font-size1);
    color: var(--jp-ui-font-color0);
    border: var(--jp-border-width) solid var(--jp-border-color1);
  }
  
  .jp-KeySelector label,
  .jp-MetadataEditorTool label {
    line-height: 1.4;
  }
  
  .jp-NotebookTools .jp-select-wrapper {
    margin-top: 4px;
    margin-bottom: 0px;
  }
  
  .jp-NotebookTools .jp-Collapse {
    margin-top: 16px;
  }
  
  /*-----------------------------------------------------------------------------
  | Presentation Mode (.jp-mod-presentationMode)
  |----------------------------------------------------------------------------*/
  
  .jp-mod-presentationMode .jp-Notebook {
    --jp-content-font-size1: var(--jp-content-presentation-font-size1);
    --jp-code-font-size: var(--jp-code-presentation-font-size);
  }
  
  .jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
  .jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
    flex: 0 0 110px;
  }
  
  </style> -->
  
      <style type="text/css">
  /*-----------------------------------------------------------------------------
  | Copyright (c) Jupyter Development Team.
  | Distributed under the terms of the Modified BSD License.
  |----------------------------------------------------------------------------*/
  
  /*
  The following CSS variables define the main, public API for styling JupyterLab.
  These variables should be used by all plugins wherever possible. In other
  words, plugins should not define custom colors, sizes, etc unless absolutely
  necessary. This enables users to change the visual theme of JupyterLab
  by changing these variables.
  
  Many variables appear in an ordered sequence (0,1,2,3). These sequences
  are designed to work well together, so for example, `--jp-border-color1` should
  be used with `--jp-layout-color1`. The numbers have the following meanings:
  
  * 0: super-primary, reserved for special emphasis
  * 1: primary, most important under normal situations
  * 2: secondary, next most important under normal situations
  * 3: tertiary, next most important under normal situations
  
  Throughout JupyterLab, we are mostly following principles from Google's
  Material Design when selecting colors. We are not, however, following
  all of MD as it is not optimized for dense, information rich UIs.
  */
  
  :root {
    /* Elevation
     *
     * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
     *
     * https://github.com/material-components/material-components-web
     * https://material-components-web.appspot.com/elevation.html
     */
  
    --jp-shadow-base-lightness: 0;
    --jp-shadow-umbra-color: rgba(
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      0.2
    );
    --jp-shadow-penumbra-color: rgba(
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      0.14
    );
    --jp-shadow-ambient-color: rgba(
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      0.12
    );
    --jp-elevation-z0: none;
    --jp-elevation-z1: 0px 2px 1px -1px var(--jp-shadow-umbra-color),
      0px 1px 1px 0px var(--jp-shadow-penumbra-color),
      0px 1px 3px 0px var(--jp-shadow-ambient-color);
    --jp-elevation-z2: 0px 3px 1px -2px var(--jp-shadow-umbra-color),
      0px 2px 2px 0px var(--jp-shadow-penumbra-color),
      0px 1px 5px 0px var(--jp-shadow-ambient-color);
    --jp-elevation-z4: 0px 2px 4px -1px var(--jp-shadow-umbra-color),
      0px 4px 5px 0px var(--jp-shadow-penumbra-color),
      0px 1px 10px 0px var(--jp-shadow-ambient-color);
    --jp-elevation-z6: 0px 3px 5px -1px var(--jp-shadow-umbra-color),
      0px 6px 10px 0px var(--jp-shadow-penumbra-color),
      0px 1px 18px 0px var(--jp-shadow-ambient-color);
    --jp-elevation-z8: 0px 5px 5px -3px var(--jp-shadow-umbra-color),
      0px 8px 10px 1px var(--jp-shadow-penumbra-color),
      0px 3px 14px 2px var(--jp-shadow-ambient-color);
    --jp-elevation-z12: 0px 7px 8px -4px var(--jp-shadow-umbra-color),
      0px 12px 17px 2px var(--jp-shadow-penumbra-color),
      0px 5px 22px 4px var(--jp-shadow-ambient-color);
    --jp-elevation-z16: 0px 8px 10px -5px var(--jp-shadow-umbra-color),
      0px 16px 24px 2px var(--jp-shadow-penumbra-color),
      0px 6px 30px 5px var(--jp-shadow-ambient-color);
    --jp-elevation-z20: 0px 10px 13px -6px var(--jp-shadow-umbra-color),
      0px 20px 31px 3px var(--jp-shadow-penumbra-color),
      0px 8px 38px 7px var(--jp-shadow-ambient-color);
    --jp-elevation-z24: 0px 11px 15px -7px var(--jp-shadow-umbra-color),
      0px 24px 38px 3px var(--jp-shadow-penumbra-color),
      0px 9px 46px 8px var(--jp-shadow-ambient-color);
  
    /* Borders
     *
     * The following variables, specify the visual styling of borders in JupyterLab.
     */
  
    --jp-border-width: 1px;
    --jp-border-color0: var(--md-grey-400);
    --jp-border-color1: var(--md-grey-400);
    --jp-border-color2: var(--md-grey-300);
    --jp-border-color3: var(--md-grey-200);
    --jp-border-radius: 2px;
  
    /* UI Fonts
     *
     * The UI font CSS variables are used for the typography all of the JupyterLab
     * user interface elements that are not directly user generated content.
     *
     * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
     * is applied to a parent element. When children elements, such as headings, are sized
     * in em all things will be computed relative to that body size.
     */
  
    --jp-ui-font-scale-factor: 1.2;
    --jp-ui-font-size0: 0.83333em;
    --jp-ui-font-size1: 13px; /* Base font size */
    --jp-ui-font-size2: 1.2em;
    --jp-ui-font-size3: 1.44em;
  
    --jp-ui-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
      Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
  
    /*
     * Use these font colors against the corresponding main layout colors.
     * In a light theme, these go from dark to light.
     */
  
    /* Defaults use Material Design specification */
    --jp-ui-font-color0: rgba(0, 0, 0, 1);
    --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
    --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
    --jp-ui-font-color3: rgba(0, 0, 0, 0.38);
  
    /*
     * Use these against the brand/accent/warn/error colors.
     * These will typically go from light to darker, in both a dark and light theme.
     */
  
    --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
    --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
    --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
    --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);
  
    /* Content Fonts
     *
     * Content font variables are used for typography of user generated content.
     *
     * The font sizing here is done assuming that the body font size of --jp-content-font-size1
     * is applied to a parent element. When children elements, such as headings, are sized
     * in em all things will be computed relative to that body size.
     */
  
    --jp-content-line-height: 1.6;
    --jp-content-font-scale-factor: 1.2;
    --jp-content-font-size0: 0.83333em;
    --jp-content-font-size1: 14px; /* Base font size */
    --jp-content-font-size2: 1.2em;
    --jp-content-font-size3: 1.44em;
    --jp-content-font-size4: 1.728em;
    --jp-content-font-size5: 2.0736em;
  
    /* This gives a magnification of about 125% in presentation mode over normal. */
    --jp-content-presentation-font-size1: 17px;
  
    --jp-content-heading-line-height: 1;
    --jp-content-heading-margin-top: 1.2em;
    --jp-content-heading-margin-bottom: 0.8em;
    --jp-content-heading-font-weight: 500;
  
    /* Defaults use Material Design specification */
    --jp-content-font-color0: rgba(0, 0, 0, 1);
    --jp-content-font-color1: rgba(0, 0, 0, 0.87);
    --jp-content-font-color2: rgba(0, 0, 0, 0.54);
    --jp-content-font-color3: rgba(0, 0, 0, 0.38);
  
    --jp-content-link-color: var(--md-blue-700);
  
    --jp-content-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
      Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
      'Segoe UI Symbol';
  
    /*
     * Code Fonts
     *
     * Code font variables are used for typography of code and other monospaces content.
     */
  
    --jp-code-font-size: 13px;
    --jp-code-line-height: 1.3077; /* 17px for 13px base */
    --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
    --jp-code-font-family-default: Menlo, Consolas, 'DejaVu Sans Mono', monospace;
    --jp-code-font-family: var(--jp-code-font-family-default);
  
    /* This gives a magnification of about 125% in presentation mode over normal. */
    --jp-code-presentation-font-size: 16px;
  
    /* may need to tweak cursor width if you change font size */
    --jp-code-cursor-width0: 1.4px;
    --jp-code-cursor-width1: 2px;
    --jp-code-cursor-width2: 4px;
  
    /* Layout
     *
     * The following are the main layout colors use in JupyterLab. In a light
     * theme these would go from light to dark.
     */
  
    --jp-layout-color0: white;
    --jp-layout-color1: white;
    --jp-layout-color2: var(--md-grey-200);
    --jp-layout-color3: var(--md-grey-400);
    --jp-layout-color4: var(--md-grey-600);
  
    /* Inverse Layout
     *
     * The following are the inverse layout colors use in JupyterLab. In a light
     * theme these would go from dark to light.
     */
  
    --jp-inverse-layout-color0: #111111;
    --jp-inverse-layout-color1: var(--md-grey-900);
    --jp-inverse-layout-color2: var(--md-grey-800);
    --jp-inverse-layout-color3: var(--md-grey-700);
    --jp-inverse-layout-color4: var(--md-grey-600);
  
    /* Brand/accent */
  
    --jp-brand-color0: var(--md-blue-700);
    --jp-brand-color1: var(--md-blue-500);
    --jp-brand-color2: var(--md-blue-300);
    --jp-brand-color3: var(--md-blue-100);
    --jp-brand-color4: var(--md-blue-50);
  
    --jp-accent-color0: var(--md-green-700);
    --jp-accent-color1: var(--md-green-500);
    --jp-accent-color2: var(--md-green-300);
    --jp-accent-color3: var(--md-green-100);
  
    /* State colors (warn, error, success, info) */
  
    --jp-warn-color0: var(--md-orange-700);
    --jp-warn-color1: var(--md-orange-500);
    --jp-warn-color2: var(--md-orange-300);
    --jp-warn-color3: var(--md-orange-100);
  
    --jp-error-color0: var(--md-red-700);
    --jp-error-color1: var(--md-red-500);
    --jp-error-color2: var(--md-red-300);
    --jp-error-color3: var(--md-red-100);
  
    --jp-success-color0: var(--md-green-700);
    --jp-success-color1: var(--md-green-500);
    --jp-success-color2: var(--md-green-300);
    --jp-success-color3: var(--md-green-100);
  
    --jp-info-color0: var(--md-cyan-700);
    --jp-info-color1: var(--md-cyan-500);
    --jp-info-color2: var(--md-cyan-300);
    --jp-info-color3: var(--md-cyan-100);
  
    /* Cell specific styles */
  
    --jp-cell-padding: 5px;
  
    --jp-cell-collapser-width: 8px;
    --jp-cell-collapser-min-height: 20px;
    --jp-cell-collapser-not-active-hover-opacity: 0.6;
  
    --jp-cell-editor-background: var(--md-grey-100);
    --jp-cell-editor-border-color: var(--md-grey-300);
    --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
    --jp-cell-editor-active-background: var(--jp-layout-color0);
    --jp-cell-editor-active-border-color: var(--jp-brand-color1);
  
    --jp-cell-prompt-width: 64px;
    --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
    --jp-cell-prompt-letter-spacing: 0px;
    --jp-cell-prompt-opacity: 1;
    --jp-cell-prompt-not-active-opacity: 0.5;
    --jp-cell-prompt-not-active-font-color: var(--md-grey-700);
    /* A custom blend of MD grey and blue 600
     * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
    --jp-cell-inprompt-font-color: #307fc1;
    /* A custom blend of MD grey and orange 600
     * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
    --jp-cell-outprompt-font-color: #bf5b3d;
  
    /* Notebook specific styles */
  
    --jp-notebook-padding: 10px;
    --jp-notebook-select-background: var(--jp-layout-color1);
    --jp-notebook-multiselected-color: var(--md-blue-50);
  
    /* The scroll padding is calculated to fill enough space at the bottom of the
    notebook to show one single-line cell (with appropriate padding) at the top
    when the notebook is scrolled all the way to the bottom. We also subtract one
    pixel so that no scrollbar appears if we have just one single-line cell in the
    notebook. This padding is to enable a 'scroll past end' feature in a notebook.
    */
    --jp-notebook-scroll-padding: calc(
      100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
        var(--jp-code-padding) - var(--jp-cell-padding) - 1px
    );
  
    /* Rendermime styles */
  
    --jp-rendermime-error-background: #fdd;
    --jp-rendermime-table-row-background: var(--md-grey-100);
    --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);
  
    /* Dialog specific styles */
  
    --jp-dialog-background: rgba(0, 0, 0, 0.25);
  
    /* Console specific styles */
  
    --jp-console-padding: 10px;
  
    /* Toolbar specific styles */
  
    --jp-toolbar-border-color: var(--jp-border-color1);
    --jp-toolbar-micro-height: 8px;
    --jp-toolbar-background: var(--jp-layout-color1);
    --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
    --jp-toolbar-header-margin: 4px 4px 0px 4px;
    --jp-toolbar-active-background: var(--md-grey-300);
  
    /* Input field styles */
  
    --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
    --jp-input-active-background: var(--jp-layout-color1);
    --jp-input-hover-background: var(--jp-layout-color1);
    --jp-input-background: var(--md-grey-100);
    --jp-input-border-color: var(--jp-border-color1);
    --jp-input-active-border-color: var(--jp-brand-color1);
    --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);
  
    /* General editor styles */
  
    --jp-editor-selected-background: #d9d9d9;
    --jp-editor-selected-focused-background: #d7d4f0;
    --jp-editor-cursor-color: var(--jp-ui-font-color0);
  
    /* Code mirror specific styles */
  
    --jp-mirror-editor-keyword-color: #008000;
    --jp-mirror-editor-atom-color: #88f;
    --jp-mirror-editor-number-color: #080;
    --jp-mirror-editor-def-color: #00f;
    --jp-mirror-editor-variable-color: var(--md-grey-900);
    --jp-mirror-editor-variable-2-color: #05a;
    --jp-mirror-editor-variable-3-color: #085;
    --jp-mirror-editor-punctuation-color: #05a;
    --jp-mirror-editor-property-color: #05a;
    --jp-mirror-editor-operator-color: #aa22ff;
    --jp-mirror-editor-comment-color: #408080;
    --jp-mirror-editor-string-color: #ba2121;
    --jp-mirror-editor-string-2-color: #708;
    --jp-mirror-editor-meta-color: #aa22ff;
    --jp-mirror-editor-qualifier-color: #555;
    --jp-mirror-editor-builtin-color: #008000;
    --jp-mirror-editor-bracket-color: #997;
    --jp-mirror-editor-tag-color: #170;
    --jp-mirror-editor-attribute-color: #00c;
    --jp-mirror-editor-header-color: blue;
    --jp-mirror-editor-quote-color: #090;
    --jp-mirror-editor-link-color: #00c;
    --jp-mirror-editor-error-color: #f00;
    --jp-mirror-editor-hr-color: #999;
  
    /* Vega extension styles */
  
    --jp-vega-background: white;
  
    /* Sidebar-related styles */
  
    --jp-sidebar-min-width: 250px;
  
    /* Search-related styles */
  
    --jp-search-toggle-off-opacity: 0.5;
    --jp-search-toggle-hover-opacity: 0.8;
    --jp-search-toggle-on-opacity: 1;
    --jp-search-selected-match-background-color: rgb(245, 200, 0);
    --jp-search-selected-match-color: black;
    --jp-search-unselected-match-background-color: var(
      --jp-inverse-layout-color0
    );
    --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);
  
    /* Icon colors that work well with light or dark backgrounds */
    --jp-icon-contrast-color0: var(--md-purple-600);
    --jp-icon-contrast-color1: var(--md-green-600);
    --jp-icon-contrast-color2: var(--md-pink-600);
    --jp-icon-contrast-color3: var(--md-blue-600);
  }
  </style>
  
  <style type="text/css">
  a.anchor-link {
     display: none;
  }
  .highlight  {
      margin: 0.4em;
  }
  
  /* Input area styling */
  .jp-InputArea {
      overflow: hidden;
  }
  
  .jp-InputArea-editor {
      overflow: hidden;
  }
  
  @media print {
    body {
      margin: 0;
    }
  }
  </style>
  
  <body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">

<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 id="Getting-Started-with-FLORIS-v3">Getting Started with FLORIS v3<a class="anchor-link" href="#Getting-Started-with-FLORIS-v3">&#182;</a></h1><p>FLORIS is a command-line program written in Python. There are two primary packages that make up the software:</p>
<ul>
<li><code>floris.simulation</code>: simulation framework including wake model definitions</li>
<li><code>floris.tools</code>: utilities for pre and post processing as well as driving the simulation</li>
</ul>
<p>Users of FLORIS will develop a Python script with the following sequence of steps:</p>
<ol>
<li>preprocess</li>
<li>calculation</li>
<li>postprocess</li>
</ol>
<p>Generally, users will only interact with <code>floris.tools</code> and most often through the <code>FlorisInterface</code> class. Additionally, <code>floris.tools</code> contains functionality for comparing results, creating visualizations, and developing optimization cases.</p>
<p><strong>NOTE <code>floris.tools</code> is under active design and development. The API's will change and additional functionality from FLORIS v2 will be included in upcoming releases.</strong></p>
<p>This notebook steps through the basic ideas and operations of FLORIS while showing realistic uses and expected behavior.</p>

</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Initialize-FlorisInterface">Initialize FlorisInterface<a class="anchor-link" href="#Initialize-FlorisInterface">&#182;</a></h2><p>The <code>FlorisInterface</code> provides functionality to build a wind farm representation and drive the simulation. This object is created (instantiated) by passing the path to a FLORIS input file. Once this object is created, it can immediately be used to inspect the data.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">floris.tools</span> <span class="kn">import</span> <span class="n">FlorisInterface</span>

<span class="n">fi</span> <span class="o">=</span> <span class="n">FlorisInterface</span><span class="p">(</span><span class="s2">&quot;inputs/gch.yaml&quot;</span><span class="p">)</span>
<span class="n">x</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">get_turbine_layout</span><span class="p">()</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;     x       y&quot;</span><span class="p">)</span>
<span class="k">for</span> <span class="n">_x</span><span class="p">,</span> <span class="n">_y</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">_x</span><span class="si">:</span><span class="s2">6.1f</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">_y</span><span class="si">:</span><span class="s2">6.1f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>     x       y
   0.0,    0.0
 630.0,    0.0
1260.0,    0.0
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Build-the-model">Build the model<a class="anchor-link" href="#Build-the-model">&#182;</a></h2><p>At this point, FLORIS has been initialized with the data defined in the input file. However, it is often simpler to define a basic configuration in the input file as a starting point and then make modifications in the Python script.
This allows for generating data algorithmically or loading data from a data file. Modifications to the wind farm representation are handled through the <code>FlorisInterface.reinitialize()</code> function with keyword arguments. Another way to
think of this function is that it changes the value of inputs specified in the input file.</p>
<p>Let's change the location of turbines in the wind farm.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Design a wind farm with turbines in a 2x2 pattern</span>
<span class="c1"># Units are in meters</span>
<span class="n">x_2x2</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">800</span><span class="p">,</span> <span class="mi">800</span><span class="p">]</span>
<span class="n">y_2x2</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">400</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">400</span><span class="p">]</span>
<span class="n">fi</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span> <span class="n">layout</span><span class="o">=</span><span class="p">(</span><span class="n">x_2x2</span><span class="p">,</span> <span class="n">y_2x2</span><span class="p">)</span> <span class="p">)</span>

<span class="n">x</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">get_turbine_layout</span><span class="p">()</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;     x       y&quot;</span><span class="p">)</span>
<span class="k">for</span> <span class="n">_x</span><span class="p">,</span> <span class="n">_y</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">_x</span><span class="si">:</span><span class="s2">6.1f</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">_y</span><span class="si">:</span><span class="s2">6.1f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>     x       y
   0.0,    0.0
   0.0,  400.0
 800.0,    0.0
 800.0,  400.0
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Additionally, we can change the wind speeds and wind directions. These are given as lists of wind speeds and wind directions that will be
expanded so that a wake calculation will happen for every wind direction with each speed.</p>
<p>Notice that we can give <code>FlorisInterface.reinitialize()</code> multiple keyword arguments at once.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># One wind direction and one speed -&gt; one atmospheric condition</span>
<span class="n">fi</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span> <span class="n">wind_directions</span><span class="o">=</span><span class="p">[</span><span class="mf">270.0</span><span class="p">],</span> <span class="n">wind_speeds</span><span class="o">=</span><span class="p">[</span><span class="mf">8.0</span><span class="p">]</span> <span class="p">)</span>

<span class="c1"># Two wind directions and one speed -&gt; two atmospheric conditions</span>
<span class="n">fi</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span> <span class="n">wind_directions</span><span class="o">=</span><span class="p">[</span><span class="mf">270.0</span><span class="p">,</span> <span class="mf">280.0</span><span class="p">],</span> <span class="n">wind_speeds</span><span class="o">=</span><span class="p">[</span><span class="mf">8.0</span><span class="p">]</span> <span class="p">)</span>

<span class="c1"># Two wind directions and two speeds -&gt; four atmospheric conditions</span>
<span class="n">fi</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span> <span class="n">wind_directions</span><span class="o">=</span><span class="p">[</span><span class="mf">270.0</span><span class="p">,</span> <span class="mf">280.0</span><span class="p">],</span> <span class="n">wind_speeds</span><span class="o">=</span><span class="p">[</span><span class="mf">8.0</span><span class="p">,</span> <span class="mf">9.0</span><span class="p">]</span> <span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><code>FlorisInterface.reinitialize()</code> creates all of the basic data structures required for the simulation but it does not do any aerodynamic calculations.
The low level data structures have a complex shape that enables faster computations. Specifically, most data is structured as a many-dimensional Numpy array
with the following dimensions:</p>
<ul>
<li>0: wind directions</li>
<li>1: wind speeds</li>
<li>2: turbines</li>
<li>3: grid-1</li>
<li>4: grid-2</li>
</ul>
<p>For example, we can see the overall shape of the data structure for the grid point x-coordinates for the all turbines and get the x-coordinates of grid points for the
third turbine in the first wind direction and first wind speed. We can also plot all the grid points in space to get an idea of the overall form of our grid.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Dimensions of grid x-components&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span> <span class="n">np</span><span class="o">.</span><span class="n">shape</span><span class="p">(</span><span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">x</span><span class="p">)</span> <span class="p">)</span>

<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Turbine 3 grid x-components for first wind direction and first wind speed&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:])</span>

<span class="n">x</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:,</span> <span class="p">:]</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">y</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:,</span> <span class="p">:]</span>
<span class="n">z</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">z</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:,</span> <span class="p">:]</span>

<span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">fig</span><span class="o">.</span><span class="n">add_subplot</span><span class="p">(</span><span class="mi">111</span><span class="p">,</span> <span class="n">projection</span><span class="o">=</span><span class="s2">&quot;3d&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">z</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;.&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_zlim</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">150</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Dimensions of grid x-components
(2, 2, 4, 3, 3)

Turbine 3 grid x-components for first wind direction and first wind speed
[[800. 800. 800.]
 [800. 800. 800.]
 [800. 800. 800.]]
</pre>
</div>
</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>(0.0, 150.0)</pre>
</div>

</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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yfEr0LFarThw4ACrekuBxdoUzMQbKeHDIhiGwYkTJ/D000/jr3/9K15++WUolUp861vfwmc+8xk8+uij6O/vR4iHd1yvx2g0gsfj4dJLL0VVVRXMZrNfgY7NZsMXv/hF3HjjjbjtttuScn0pfIo1saJzMTY2hl27dqGrqwtlZWUwGAwAFleX7OxsGAwGXHPNNXjggQewc+dOAIuh7k9/+tNkFVZWBFqtFi+//DJefPFFzM7OevXUJ2qln5qaglqtRnNz87KmE2omqVKp8MMf/hAMw+Bzn/scHnzwwXWxbx0hUis6FynhQ2AolUrccccdOHbsGP73f/8XGzduxL/+679i586deOihh3DmzBl4PJ64nU+lUgUkOQDWrnnHjh3Iz89Hfn4+3n77bdx0000xnff2229HXl4eaxEGADqdDldccQVqa2txxRVXQK/XA1h8+N9zzz2oqalBU1MTTp8+HdO51zLWDNGDCR8ArBvhQziQy+W45ZZb8PLLL+Pdd99FW1sbfvazn+Hiiy/Ggw8+iJMnT8ZE+nPnzmF2djYgySlcLhfuvPNObN++HS+++CL+8pe/4Omnn476vADwpS99Ca+//rrXa4888gguu+wyDA4O4rLLLmM73V577TUMDg5icHAQ//3f/4277rorpnOvZawJohNCcMcdd6Curg7f+ta32Nf37t2LZ555BgDwzDPPYN++fezrv/vd70AIwYcffoisrKzzQq8cDTIyMnDo0CG88MILOHHiBC6++GL86le/wkUXXYT7778fH3zwAdxud9jHo4XRUCR3u9246667UF9fj+9+97tstBXrdNpdu3YtExOtt1pNVCCEBPtvVeC9994jAEhjYyNpbm4mzc3N5NixY2Rubo7s3r2b1NTUkMsuu4xotVpCCCEej4d87WtfI1VVVaShoYGcPHlyhd/B6oPVaiWvvvoqueWWW0hDQwP58pe/TF577TUyPz9PzGaz3/+GhobIu+++SxYWFgL+jNlsJgsLC+RLX/oSeeCBB4jH44n7tY+OjpLNmzezX2dlZbH/9ng87NdXX301ee+999jv7d69e7XeC6F4GPN/a6KpZefOnQEryW+99day1xiGwS9+8YtEX9aahkQiwZ49e7Bnzx44HA785S9/wZEjR3Dfffdhx44d2L9/Py655BK2p35mZoa1iA62kns8Hnz7299GdnY2Hn744aTXTdZ7rSYQ1gTRkw23242tW7eiuLgYR48eTcbY3BWFSCTCVVddhauuugoulwvvvvsuXnjhBXz3u99FW1sb8vPzYTQa8eijjwY1sPB4PHjggQcgEonw2GOPJc2fbj02qUSKNZGjJxtPPPEE6urq2K/vv/9+fPOb38TQ0BCys7Px1FNPAQCeeuopZGdnY2hoCN/85jdx//33r9Qlxw0CgQC7d+/Gk08+ibNnz2LDhg149tln8dFHH+ErX/kKXn31VVgslmW/5/F48IMf/AAOhwP/8R//kVQTylStJgyEiO3XHSYnJ8nu3bvJW2+9Ra6++mri8XiIUqkkTqeTEELIBx98QD772c8SQgj57Gc/Sz744ANCCCFOp5MolcqE5KQrBafTSW677TZiMBiI2+0mJ06cIN/61rdIU1MTuf7668nvf/97Mjs7S0wmE7n//vvJrbfeSlwuV0Kv6dChQ6SgoIAIBAJSXFxMfv3rX58PtZqE5+gpovvg+uuvJ5988gl5++23ydVXX000Gg2prq5mvz8xMcEWgjZv3kwmJyfZ71VVVRGNRpP0a0423G43+eSTT8j9999PWlpaSH19Pdm/f3/CSX4eI1WMSyaOHj2KvLw8bNmyBe+8885KX86qBY/Hw5YtW7Blyxb8+Mc/xtGjR7F79+7z3ZJ5TSNFdA7ef/99vPrqqzh+/DhsNhsWFhZw7733nhdjcxMFHo+HvXv3rvRlpBACqWIcBz/5yU8wNTWFsbExPPvss9i9ezf+8Ic/4DOf+QwOHz4MYHmxhxaBDh8+jN27d6e2dlJYlUgRPQz89Kc/xeOPP46amhpotVrccccdAIA77rgDWq0WNTU1ePzxx6MaMpBCCsnAmuteS2F94Hye9OMHqe618xkGgwE33HADNm3ahLq6Opw4cSLViYVFwdLXv/51vPbaa+jp6cGf/vQn9PT0rPRlrWmkiL6CuPfee3HVVVehr68PZ8+eRV1dXaoTC+tr0k+ykCL6CmF+fh7vvvsum++LRCLI5fJUJxYCT0xJIXqkiL5CGB0dRW5uLm677Ta0trbizjvvhNlsjnhcUAophIMU0VcILpcLp0+fxl133YX29nakp6cvq9qv106sVDNK/JEi+gqhpKQEJSUl2LFjBwDghhtuwOnTp1OuOQC2bduGwcFBjI6OwuFw4Nlnn02JcmJEiugrhIKCApSWlqK/vx/AYl99fX19qhMLix10//Vf/4Urr7wSdXV1OHDgQGqcV4xI7aOvIM6cOYM777wTDocDVVVVePrpp+HxeFbbuKAUEo/UAIcUUlgHSDjRQzW1rL9K0HkKhmG+CeBOLD68OwHcBqAQwLMAlABOAbiZEOJgGEYM4HcAtgDQAjhICBlbietOIT5I5ejrAAzDFAO4B8BWQkgDAD6AQwB+CuBnhJAaAHoAdyz9yh0A9Euv/2zp51JYw0gRff1AACCNYRgBACmAaQC7ARxe+v4zAPYv/Xvf0tdY+v5lzHrc5zuPkCL6OgAhRAXgMQATWCT4PBZDdQMhxLX0Y1MA6H5dMYDJpd91Lf38+mq0P8+QIvo6AMMw2VhcpSsBFAFIB3DVil5UCklFiujrA5cDGCWEaAghTgAvArgYgHwplAeAEgBUU6sCUAoAS9/PwmJRLoU1ihTR1wcmAFzAMIx0Kde+DEAPgLcB3LD0M7cCoC1iry59jaXv/4WE2IdNYXUj1D56CucJGIb5EYCDAFwA2rG41VaMxe01xdJr/x8hxM4wjATA7wG0AtABOEQIGVmRC08hLkgRPYUU1gFSoXsKKawDpIieQgrrACmip5DCOkCK6CmksA6QInoKKawDpIieQgrrACmip5DCOkCK6CmksA7w/wPWhJu78Lh/JAAAAABJRU5ErkJggg=="
class="
jp-needs-light-background
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Execute-wake-calculation">Execute wake calculation<a class="anchor-link" href="#Execute-wake-calculation">&#182;</a></h2><p>Running the wake calculation is a one-liner. This will calculate the velocities at each turbine given the wake of other turbines for every wind speed and wind direction combination.
Since we have not explicitly specified yaw control settings, all turbines are aligned with the inflow.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fi</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Get-turbine-power">Get turbine power<a class="anchor-link" href="#Get-turbine-power">&#182;</a></h2><p>At this point, the simulation has completed and we can use the <code>FlorisInterface</code> to extract useful information such as the power produced at each turbine. Remember that we have configured the simulation with two wind directions, two wind speeds, and four turbines.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">powers</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">get_turbine_powers</span><span class="p">()</span> <span class="o">/</span> <span class="mf">1000.0</span>  <span class="c1"># calculated in Watts, so convert to kW</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Dimensions of `powers`&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span> <span class="n">np</span><span class="o">.</span><span class="n">shape</span><span class="p">(</span><span class="n">powers</span><span class="p">)</span> <span class="p">)</span>

<span class="nb">print</span><span class="p">()</span>
<span class="c1"># TODO hold wind direction fixed and loop over wind speeds</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Turbine powers for 8 m/s&quot;</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">2</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Wind direction </span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">powers</span><span class="p">[</span><span class="n">i</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:])</span>

<span class="c1"># TODO: maybe describe get_farm_power() here?</span>

<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Turbine powers for all turbines at all wind conditions&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">powers</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Dimensions of `powers`
(2, 2, 4)

Turbine powers for 8 m/s
Wind direction 0
[1691.32664838 1691.32664838  592.6531181   592.97842923]
Wind direction 1
[1691.32664838 1691.32664838 1631.06744171 1629.75543674]

Turbine powers for all turbines at all wind conditions
[[[1691.32664838 1691.32664838  592.6531181   592.97842923]
  [2407.84167188 2407.84167188  861.30649817  861.73255027]]

 [[1691.32664838 1691.32664838 1631.06744171 1629.75543674]
  [2407.84167188 2407.84167188 2321.41247863 2319.53218301]]]
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Applying-yaw-angles">Applying yaw angles<a class="anchor-link" href="#Applying-yaw-angles">&#182;</a></h2><p>Yaw angles are applied to turbines through the <code>FlorisInterface.calculate_wake</code> function.</p>
<p><strong>Note that <code>yaw_angles</code> is a array</strong> -- You must provide yaw angles in a array with dimensions equal to:</p>
<ul>
<li>0: number of wind directions</li>
<li>1: number of wind speeds</li>
<li>2: number of turbines</li>
</ul>
<p><strong>Unlike data set in <code>FlorisInterface.reinitialize()</code>, yaw angles are not stored in memory and must be given again in successive calls to <code>FlorisInterface.calculate_wake</code>.</strong>
<strong>If no yaw angles are given, all turbines will be aligned with the inflow.</strong></p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Array of zero yaw angles</span>
<span class="n">yaw_angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span> <span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">4</span><span class="p">)</span> <span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">yaw_angles</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]]]
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Yaw the front row for all wind directions and all wind speeds</span>
<span class="n">yaw_angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span> <span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">4</span><span class="p">)</span> <span class="p">)</span>
<span class="n">yaw_angles</span><span class="p">[:,</span> <span class="p">:,</span> <span class="mi">0</span><span class="p">:</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span> <span class="mi">25</span>

<span class="c1"># TODO Pass the settings to Floris</span>
<span class="n">fi</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">(</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span> <span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">yaw_angles</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>[[[25. 25.  0.  0.]
  [25. 25.  0.  0.]]

 [[25. 25.  0.  0.]
  [25. 25.  0.  0.]]]
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Start-to-finish">Start to finish<a class="anchor-link" href="#Start-to-finish">&#182;</a></h2><p>Let's put it all together. The following code does the following:</p>
<ol>
<li>Load an input file</li>
<li>Modify the inputs with a more complex wind turbine layout</li>
<li>Change the wind speeds and wind directions</li>
<li>Execute the simulation</li>
<li>Get the total farm power</li>
<li>Add yaw settings for some turbines</li>
<li>Execute the simulation</li>
<li>Get the total farm power and compare to without yaw control</li>
</ol>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">floris.tools</span> <span class="kn">import</span> <span class="n">FlorisInterface</span>

<span class="n">fi</span> <span class="o">=</span> <span class="n">FlorisInterface</span><span class="p">(</span><span class="s2">&quot;inputs/gch.yaml&quot;</span><span class="p">)</span>

<span class="c1"># Construct the model</span>
<span class="n">D</span> <span class="o">=</span> <span class="mf">126.0</span>  <span class="c1"># Design the layout based on turbine diameter</span>
<span class="n">x</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span> <span class="o">*</span> <span class="n">D</span><span class="p">,</span>  <span class="mi">6</span> <span class="o">*</span> <span class="n">D</span><span class="p">,</span> <span class="mi">9</span> <span class="o">*</span> <span class="n">D</span><span class="p">]</span>
<span class="n">y</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span> <span class="o">*</span> <span class="n">D</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span> <span class="o">*</span> <span class="n">D</span><span class="p">,</span> <span class="mi">3</span> <span class="o">*</span> <span class="n">D</span><span class="p">]</span>
<span class="n">wind_directions</span> <span class="o">=</span> <span class="p">[</span><span class="mf">210.0</span><span class="p">,</span> <span class="mf">270.0</span><span class="p">]</span>
<span class="n">wind_speeds</span> <span class="o">=</span> <span class="p">[</span><span class="mf">8.0</span><span class="p">]</span>

<span class="c1"># Pass the new data to FlorisInterface</span>
<span class="n">fi</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span>
    <span class="n">layout</span><span class="o">=</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">),</span>
    <span class="n">wind_directions</span><span class="o">=</span><span class="n">wind_directions</span><span class="p">,</span>
    <span class="n">wind_speeds</span><span class="o">=</span><span class="n">wind_speeds</span>
<span class="p">)</span>

<span class="c1"># Calculate the velocities at each turbine for all atmospheric conditions with no yaw control settings</span>
<span class="n">fi</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">()</span>

<span class="c1"># Get the farm power</span>
<span class="n">turbine_powers</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">get_turbine_powers</span><span class="p">()</span> <span class="o">/</span> <span class="mf">1000.0</span>
<span class="n">farm_power_baseline</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">turbine_powers</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>

<span class="c1"># Develop the yaw control settings</span>
<span class="n">yaw_angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span> <span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">4</span><span class="p">)</span> <span class="p">)</span>  <span class="c1"># Construct the yaw array with dimensions for two wind directions, one wind speed, and four turbines</span>
<span class="n">yaw_angles</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="o">-</span><span class="mi">5</span>            <span class="c1"># At 210 degrees, yaw the first turbine 10 degrees</span>
<span class="n">yaw_angles</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="o">-</span><span class="mi">20</span>           <span class="c1"># At 210 degrees, yaw the second turbine 10 degrees</span>
<span class="n">yaw_angles</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="p">:,</span> <span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="mi">20</span>            <span class="c1"># At 270 degrees, yaw the second turbine 10 degrees</span>

<span class="c1"># Calculate the velocities at each turbine for all atmospheric conditions given the yaw control settings</span>
<span class="n">fi</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">(</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span> <span class="p">)</span>

<span class="c1"># Get the farm power</span>
<span class="n">turbine_powers</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">get_turbine_powers</span><span class="p">()</span> <span class="o">/</span> <span class="mf">1000.0</span>
<span class="n">farm_power_yaw</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">turbine_powers</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>

<span class="c1"># Compare power difference with yaw</span>
<span class="n">difference</span> <span class="o">=</span> <span class="mi">100</span> <span class="o">*</span> <span class="p">(</span><span class="n">farm_power_yaw</span> <span class="o">-</span> <span class="n">farm_power_baseline</span><span class="p">)</span> <span class="o">/</span> <span class="n">farm_power_baseline</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Power </span><span class="si">% d</span><span class="s2">ifference with yaw&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;    210 degrees: </span><span class="si">{</span><span class="n">difference</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]</span><span class="si">:</span><span class="s2">4.2f</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;    270 degrees: </span><span class="si">{</span><span class="n">difference</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">]</span><span class="si">:</span><span class="s2">4.2f</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Power % difference with yaw
    210 degrees: 3.95%
    270 degrees: 2.59%
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Visualization">Visualization<a class="anchor-link" href="#Visualization">&#182;</a></h2><p>While comparing turbine and farm powers is meaningful, a picture is worth at least 1000 Watts, and the <code>FlorisInterface</code> provides powerful routines for visualization.</p>
<p><strong>NOTE <code>floris.tools</code> is under active design and development. The API's will change and additional functionality from FLORIS v2 will be included in upcoming releases.</strong></p>
<p>The visualization functions require that the user select a single atmospheric condition to plot. However, the internal data structures still have the same shape but the wind speed and wind direction
dimensions have a size of 1. This means that the yaw angle array used for plotting must have the same shape as before but a single atmospheric condition must be selected.</p>
<p>Let's create a horizontal slice of each atmospheric condition from above with and without yaw settings included. Notice that although we are plotting the conditions for two different wind directions,
the farm is rotated so that the wind is coming from the left (west) in both cases.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">floris.tools.visualization</span> <span class="kn">import</span> <span class="n">visualize_cut_plane</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>

<span class="n">fig</span><span class="p">,</span> <span class="n">axarr</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">210</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;210 - Aligned&quot;</span><span class="p">)</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">210</span><span class="p">],</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;210 - Yawed&quot;</span><span class="p">)</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">270</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;270 - Aligned&quot;</span><span class="p">)</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">270</span><span class="p">],</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span><span class="p">[</span><span class="mi">1</span><span class="p">:</span><span class="mi">2</span><span class="p">,</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;270 - Yawed&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">



<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;matplotlib.collections.QuadMesh at 0x7faf94f25ca0&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
class="
jp-needs-light-background
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>We can also create a visualization of the streamwise inflow velocities on the turbine rotor grid points located on the rotor plane.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">floris.tools.visualization</span> <span class="kn">import</span> <span class="n">plot_rotor_values</span>

<span class="n">fig</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">plot_rotor_values</span><span class="p">(</span><span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">flow_field</span><span class="o">.</span><span class="n">u</span><span class="p">,</span> <span class="n">wd_index</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">ws_index</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">n_rows</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">n_cols</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">return_fig_objects</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">fig</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s2">&quot;Wind direction 210&quot;</span><span class="p">)</span>

<span class="n">fig</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">plot_rotor_values</span><span class="p">(</span><span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">flow_field</span><span class="o">.</span><span class="n">u</span><span class="p">,</span> <span class="n">wd_index</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">ws_index</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">n_rows</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">n_cols</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">return_fig_objects</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">fig</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s2">&quot;Wind direction 270&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">



<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>Text(0.5, 0.98, &#39;Wind direction 270&#39;)</pre>
</div>

</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
class="
jp-needs-light-background
"
>
</div>

</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
class="
jp-needs-light-background
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="On-Grid-Points">On Grid Points<a class="anchor-link" href="#On-Grid-Points">&#182;</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>In FLORIS, grid points are the points in space where the wind conditions are calculated.
In a typical simulation, these are all located on a regular grid on each turbine rotor.</p>
<p>The parameter <code>turbine_grid_points</code> specifies the number of rows and columns which define the turbine grid.
In the example inputs, this value is 3 meaning there are 3 x 3 = 9 total grid points for each turbine.
Wake steering codes currently require greater values greater than 1 in order to compute gradients.
However, it is likely that a single grid point (1 x 1) is suitable for non wind farm control applications,
although retuning of some parameters could be warranted.</p>
<p>We can visualize the locations of the grid points in the current example using pyplot</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Get the grid points</span>
<span class="n">xs</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">x</span>
<span class="n">ys</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">y</span>
<span class="n">zs</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">grid</span><span class="o">.</span><span class="n">z</span>

<span class="c1"># Consider the shape</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;xs has shape: &#39;</span><span class="p">,</span> <span class="n">xs</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="s1">&#39; of 2 wd x 2 ws x 4 turbines x 3 x 3 grid points&#39;</span><span class="p">)</span>

<span class="c1"># Lets plot just one wd/ws conditions</span>
<span class="n">xs</span> <span class="o">=</span> <span class="n">xs</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:,</span> <span class="p">:]</span>
<span class="n">ys</span> <span class="o">=</span> <span class="n">ys</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:,</span> <span class="p">:]</span>
<span class="n">zs</span> <span class="o">=</span> <span class="n">zs</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:,</span> <span class="p">:]</span>

<span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">fig</span><span class="o">.</span><span class="n">add_subplot</span><span class="p">(</span><span class="mi">111</span><span class="p">,</span> <span class="n">projection</span><span class="o">=</span><span class="s2">&quot;3d&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">xs</span><span class="p">,</span> <span class="n">ys</span><span class="p">,</span> <span class="n">zs</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;.&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_zlim</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">150</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>xs has shape:  (2, 1, 4, 3, 3)  of 2 wd x 2 ws x 4 turbines x 3 x 3 grid points
</pre>
</div>
</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>(0.0, 150.0)</pre>
</div>

</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPoAAADtCAYAAACWP2geAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABlxUlEQVR4nO29eXxcdb0+/pzZJ3sme5qkaZKmbdI0adK0AoKggLJI2SmyCFpZFEFFr8BV5OoPUfTicoWLX0DRe2m50AqFUotSKLJ3b9Zm37eZzJLZ1/P5/ZF8Dmcms885SdrM83r5kk6Sc87MnOd83p/3+3k/b4YQgiSSSOLMhmSxLyCJJJIQH0miJ5HEMkCS6EkksQyQJHoSSSwDJImeRBLLAEmiJ5HEMoAsws+TtbckkhAfjNgnSK7oSSSxDJAkehJJLAMkiZ5EEssASaInkcQyQJLoSSSxDJAkehJJLAMkiZ5EEssASaInkcQyQJLoSSSxDJAkehJJLAMkiZ5EEssASaInkcQyQJLoSSSxDJAkehJJLAMkib5IYFkWSQfeJBYKkfrRkxAYhBB4vV7Y7XYAgFwuh1wuh1QqBcMwYBjRW5OTWIZgIqwqySVHQLAsC4/HA5/Ph5mZGahUKj9iSyQSyGQyjvgSSTLgWiYQ/emeJPoCgBACn88Hj8cDj8eDzs5OuN1ueDweyGQyZGVlISsrCxkZGQCAvr4+lJSUQK1WJ4m/PJAk+ukOQojfKt7Z2Yny8nLk5OSAYRi43W4YjUYYjUaYzWYoFAq4XC6Ul5cjNzfX71jJFf+MRZLopzNYloVer8fExATkcjl0Oh3q6uqgVqvhdruD7sddLhdaW1uhVCrhcDigVCqRnZ2NrKwspKam+v2uRCKBXC6HTCZLEv/0huhETybjRAA/VHe5XJiYmEBRURGam5shkUjCZtuVSiXUajXKy8uRmpoKh8MBk8mE0dFRWK1WqFQqjvgpKSlwuVxwuVwAAKlUyq32MpksmdhLgkOS6AKDEAK3282t5qdOnUJqairWrFkT1/HUajXUajWKiopACOGIPzw8DKvVipSUFI74arUaPp+P+1ur1Yrc3FxuxU8Sf/kiSXQBQVdxn8+Hvr4+WCwW1NTUYGxszO/3aBmNEBKUfPRnwV5PSUlBSkoKiouLQQiB3W6H0WjE4OAgbDYbUlNTOeKfOnUKmzZt4v6ervhJ4i8/JIkuAGht3Ov1wuFwoK2tDXl5eWhqaoLNZhNNGMMwDFJTU5GamoqSkhIQQmCz2WA0GtHf3w+Hw4FTp05xxFepVHA6ndzfJ4m/fJAkeoKgtXGWZTE1NYWBgQHU1NQgKysLQOjVORzi+Rv6d2lpaUhLS0NpaSkOHTqEsrIyGI1G9PT0wOVyIS0tjSM+TfhRgieJf+YiSfQ4wU+4+Xw+dHV1wev1orm5GXK5nPu9eEkrBBiGQXp6OtLT01FWVgaWZWG1WmE0GtHV1QW324309HSO+AzDJIl/hiJJ9DjAD9WtViva29tRUlKCkpKSeWSIlGUPBrEeDhKJBBkZGcjIyMDKlSvBsiwsFguMRiM6Ozvh8XiQmZmJ7OxsZGZmcsQfGRlBUVERJ+BJEv/0Q5LoMYIvYx0bG8PY2Bjq6uqQlpYW8m/iIfpCQCKRIDMzE5mZmSgvLwfLspiZmeHKeT6fD5mZmZiZmUFubi4UCgW8Xi/391S8I5PJIJFIksRfwkgSPUrwQ3Wv14v29nYoFAps3rwZUqk05N/Fs6LT8y00JBIJsrOzkZ2djVWrVnFqPr1ej66uLgDwW/F9Ph9HfIZh/EL9JPGXFpJEjwL82rjJZEJnZycqKytRWFgY9d/HgqVCEKlUCo1Gg5SUFKxbtw5SqRQmkwkmkwlDQ0MghCArKwvZ2dmcTp9PfBrmJ4m/+EgSPQJYluVIPjg4iOnpaTQ2NkKtVkf196fTih4K9FqkUilycnKQk5MDYJbUJpMJBoMBAwMDYBjGj/herxcejwfApys+DfeTxF9YJIkeAvyEm9vtRmtrKzIzMzkZa6zH4sNsNqOtrY3TsWdnZyMlJYW78RczUx8MoYQ9MpkMubm5XPONx+OByWTC9PQ0+vr6IJVKOeKnp6dDr9fDZDKhoqKCW/H5DTpJ4ouHJNGDgBCCmZkZjI2NISMjAz09PVi7di23ksUC/opOCMHIyAjGxsZQU1MDQoifuIXWuPkJr9MJcrkceXl5yMvLAwC43W6YTCZotVr09vaCEAKZTAaLxYL09HSubReYfbgFNugkiS8ckkQPAA03PR4PxsfHMTMzg+bmZigUiriOR1dnr9eLtrY2yOVybN68GSzLgmVZP1Wb1WqFwWCA0WiEyWSCRqPhatzxnl8IhFrRI0GhUCA/Px/5+fkAgPHxcej1ekxOTqK7uxsKhYJb8dPS0uB2u7kGHdqZl3TfEQZJos+BH6rb7Xa0tbVBIpGgsbEx4RvM6/Xi0KFDWLVqFYqKigDM7v354ItbPB4PsrKyIJPJYDQaMTo6CpZlkZmZCY1Gg8zMTMhkC/vVCUEymUyG9PR0lJeXAwCcTidMJhPGx8dhsVj8WnL5xGcYZl4vfpL4sSFJdPjLWCcmJjA0NITq6moMDQ0ldDPRUN3pdOLss8+e108eCvQmps4ztNRlMplgNBq5xBfd32dmZoraiy5UvoBlWb/rVKlUKCws5KoXDoeDe7BZLBao1Wq/Xny32w232839Lm3wSfbiR8ayJnpgbbyrqwssy2Lz5s0ghMxbdWMBP1RPSUmJmuShEJjx9ng8MBqN3P5XLpdzxE9PTxd0tYs3dI/1OLQll3bmUeIHa8kdGRlBYWEh9xBKuu+Ex7IlOt/iyWKxoL29HStXrkRxcTEYhoHX642b6DSrTkP1Dz/8MKa/jybrLpfL/fa/LpcLBoPBz6DC7XbDZrP5ZfQXE7E8MPgtuStWrJjXkmswGOByuZCXl8d15vFX/KT7jj+WJdH5tfGRkRFMTExgw4YNfjJWiUQSM9H5WfX6+vqEV/FYoFQqUVRU5GdQcfz4cQwMDMBut3N96tnZ2VFrACgWakUPh8CW3La2NhQUFMDpdKKvr8+vakGJz3ffWe7EX1ZE5yfcPB4P2tvboVarsWXLlnlffKy17MCsejhZbCQkWkenq6FCocD69eu5jL7RaER3dzdcLhcyMjI44i9URp8QIhjBCCFIS0tDXl4eSktL/d5jqJbc5Wy7tWyIzpexGo1GnDp1CqtXr+ZC30DE8sUHhupLDcHaVS0WCwwGA8bGxuDz+bgyF832BztGohAqMgh2rHhacqkJh8FgQEZGBtLS0s7YzrxlQXSacGNZFv39/TAajWhqaoJKpUrouGKF6mIr4/hda/zmFbr/5UtZMzMzBTuvkEQPzOAHIpaWXK1WC4VC4feAO9N68c9oovNDdWqjrNFo0NzcLEhtXKhQfbFBm1c0Gg2A+VJWu92OwcFBLqMfb/gdiZxiHitcS67RaITNZvMrV/JXfOD0J/4ZS3R+bVyn06G3txfr1q3jbuZEIHaovtha90Ap6yeffAKVSsUJW6jldHZ2NlJTU6O+6RdyRY8Efkuu1WrFqlWruGEaw8PDIIQENeE4Xd13zjiiB1o89fT0wOl0JiRj5R9bqFA93I2x2EQPhEQimSdsMRgMfs6zVK4bLqMv5h49EbAsC7lcjrS0NG4h8Hq93HYmWEtuIPH5LblLkfhnFNEJIVy4Zbfb0draiuLiYqxbt25Jheo+nw/d3d0ghECj0SArK+u0Cv3VajVWrFjB1bep8yx9qNKkV3Z2NpRKJfd3QpITEK5v3+fzzfv8ZTJZTC25fBMOANi9ezcuvfTSJZOcPWOITmvjH374IcrLyzE8PIz169dzhgiJQMhQ3WazoaWlBUVFRVAoFDAYDOjv7+f2ydnZ2QAWth/dxxK82alDn86O81drULci+s8s0HmWn/Tq6OiA1+vlQmCfz7fkVjogum1ArC25e/bswec+97mozs8wzJ8AXA5ASwhZP/faIwC+AUA392sPEUL2zf3sQQBfB+ADcC8h5M1I5zjtiR4YqrtcLhiNRmzevDnhxg9CCIaHhwXLqk9OTqK/vx/r169HSkoKfD4fd+O43W4YDAauw0uhUMDn80Gj0UCtVotKEIPNja4pK3JS5fhwwBgT0QMRmPTiZ/S1Wi2MRiNyc3O5ve9SiGRYlo358w3XkvvNb34THR0deOaZZ3DFFVdgy5Ytke7F5wH8AcBfA17/DSHk1/wXGIapAbANQC2AYgBvMQxTTQjxIQxOa6Lza+NmsxkdHR2QyWSoq6tL+Nh0GIPFYkk4VCeEoLOzEw6Hg7OD9nq9fqGsQqHg9sEjIyPcaCWq+qLhsEajEUzgMqS3491eA4ozlMhNlWPa5kFjqXDlNMA/o+/1epGXl8eNq6IrIX1fiWT0E4EQnXD8ltydO3fi85//PNatW4e//vWvWL9+fdgyJSHkXwzDlEd5qq0AXiSEuAAMMAzTC2AzgI/C/dFpS3R+bXx4eBhTU1NoaGjAiRMnEt4L0lBdoVBg3bp1Cd18TqcTdrsdRUVFWLt2bVTXxTAMFAoFiouLUVJS4hcOt7W1cQKXRPf3b3VNw+Vhcczswo2bipGulCFTLd4tQY0nMjIy/CIZo9GIiYkJdHV1+bnupKWlLclQPxp4vV7ccsstuO222xI5zD0Mw9wK4AiA+wkhRgArAHzM+53RudfC4rQjeqCMta2tDampqdi8eTPnShIv0QOz6m1tbQmVceiQRZVKhfLy8pgaOvh79GDhME0M0f19tKui28vig34jCCHIT1egc9KGVIUU2SlypCnFvR2CfS8KhQIFBQUoKCgAgHkda4Ea/dOF+ALIff8bwM8wO7r8ZwD+E8DX4j3YaUV0fm3cYDCgq6sL1dXV3D4J+LQZJdYPOVhWPZ7GFmD2S+7v74der0dTUxNOnjwpaHItsGWVror8OrdGowHLsvPI1TJuxvu9BoABPrc6BzduykSmWnySA9Fl3QNbVe12OwwGA3p7e+F0Ojn9eiItxGJDiO+aEDJF/5thmGcA7J375xiAUt6vlsy9FhanBdH5CTcqY52ZmQkqY5VKpfD5fDEl4kJl1eNxcKVGkmlpadi0aZNflBEtGIaJ6Ubmr4q0c422cR4+fNhvf6+SSYA5rqUppSjNjq2TLRHEGmnxO9Zo4wrV6DudThw+fNivOYc/CiuWaxILiUQfDMMUEUIm5v55FYC2uf9+DcAOhmGewGwybjWAQ5GOt+SJzg/VnU4nWltbkZubi02bNgX9IGNZhSMJYGIl3MzMDNra2uY1y8R6nETA7+OemJjApk2bOHK0t7fD4/HgrNw0pGdkYn1hyoJcE0U82W0+GIZBRkYGUlNTodfrsXHjRi6jPzIy4idqiTZ3keg1BUMcD7SdAM4HkMswzCiAnwA4n2GYBsyG7oMA7pw7djvDMC8B6ADgBfCtSBl3YIkTPXBSaX9/P2pqarhaczBES/RoBDDRruj8B8bGjRuRkuJPoFhvJCGVcZQcGRkZfvt7o9GIlpMnOSnoQmS9hWpTpVszvowVmC9q4f88IyMj6LlZlhW8xOdwOGLq+SeE3Bjk5efC/P6jAB6N5ZqWJNEDQ3XaZrh58+aI4ZlUKo1I9GgFMNE8NOh4JqlUGvKBEWxFX6ykUrT7+0CveSEglDIuVA4mUNRCa9tTU1Oc6yx9qNGMvpCNNhQ0ibiUsOSIzq+N22w2tLW1YcWKFSgtLY3qJpFIJFwNOtixY9GqRwq5rVYrWltbUVZWhhUrQlc4FnNFj4RQ+/v+/n7Y7XbOldblcvnJWeOBkESP5jiBdtNOp5ML8y0WC+flRwgRVJ5rs9nCDt1cDCwpovMtnsbGxjA6Ooq6ujqkp6dHfYxQq3A8WvVwofvExAQGBgaiur6F3KMnAv7+nnrN0/09lbNGMqgIByEtqeJZhVUqlZ/dlt1ux+TkJGw2Gw4fPsxl9LOzsxPyKqA+fUsJS4Lo/ISb1+vlFG5btmyJef8UjOjxatWDEZRlWZw6dQoulytqmW08Wfel0L1G9/dKpRIbN270298PDg7GvL8XO3SPBTSjn5eXB4/HgzVr1nADNE6dOgW3242MjAxOlBSLGjG5ogcBP1SfmZlBR0dHTJNKA8HfoyfaVhr40HA4HGhpaUFBQUFMHXGBxCWEcIYOOTk583zblgrRA5Ho/l6oDLeQ+2rauca3oqKONGazmfOZj8Zui8JqtSaJzofX68Xg4CDS0tJgMBgwPT0dNGsdC+geXYi2Uj7hdDoduru7I2b9Ix3H7XajpaUFGRkZKCoqgslkwtjYGFiW5VbH0yHMB4Lv7+ksObq/p8RXKpWCZd0XwsBCIpGEHKBB7bZCDdCw2+1JogP+obrVasXIyAhycnLimlQaCIlEApvNhoGBgYTbSulDo6enByaTCZs2bYorIUVvSn6dPScnBx6PB9nZ2Vi1ahW8Xi/X4aXX68EwjN988qUu/Qzmwx64v3c6nZienkZ2dnZCnYVCr+jRHCvSAA2ZTMbJdM1mc9TR49e+9jX8+c9/1sK/RfVXAL4MwA2gD8DthBDTXONLJ4CuuT//mBByVzTnWXCi82vjdODeihUrUF1dnfCxydwUVErKSB92pJWBZVn09vYiPz8/pEAnGjAMg8nJSb+IJbAyIJPJuLZHnU4Ho9EIiUTCTVrlr44LOXAx3i1EsPr9oUOHMDMzg6GhoYTq90J7z8UT7YUaoLFnzx78/ve/R1ZWFtRqNa666iqUlJSEPM5tt92GP//5z1+Cf4vqPwE8SAjxMgzzSwAPAvjh3M/6CCENsV7vghE9sDbe19cHi8WCsrKyhMs2wKdZdbfbjdLS0ogk/9vxcbzTpcO5Vbm4rql4Honp3rO4uBirV6+O+7qovbRarY56C8Ews7PD+S4uZrOZs2fmh/kLMXdNiGiC+qdXVVUBmN+1Rn3ooolgFtNkMhToAI077rgDdrsdKpUKCoUCY2NjYYl+3nnnAYCB/xoh5B+8f34M4NpEr29BiM4ff+RwONDW1ob8/Hw0NTVxvuKJgJ9VB2aTZuHgcPvwz04t8tOVeKdbh0vWFyBdJeOudWhoCJOTkygpKUkoX0CTdwqFAhUVFXErsBiG8bNn5of5PT09UCqVXM93PGG+x8fi/T4jpsxOnFelQXGW+Pr3cPt7OnWFv7/nQ8g9ejAbqcBzdU3ZAABrCqIzwrTb7aitrcXVV18txCV+DcD/8f69imGY4wDMAH5ECHkvmoOITnRaGyeEYHJyEoODg6ipqUFWVhaA2Sc9nZcVK4Jl1bVabchkVvu4GQdO6bBlVTbWFaajc9KCyrxUpChmv2gaFSgUCmzevBmjo6Nxh660RbWmpgaTk5Pzfp6IOSQ/zAcwT+RCy0LRJvV0Vjd6dDakKaQ4MmzGFTyiC+3zFgzR7O/pyOisrCzBV/RwW6FPBk34yyejAICvbinBZ1ZFTsQKlXVnGObfMatnf2HupQkAZYQQPcMwTQBeZRimlhBijnQs0YjOT7j5fD50dXXB6/VyDisUtNssVoTKqodSxnl9LJ56dwAMA7RNmPHzrTXw+Ahy0xSQShhYLBa0traivLwcxcXFAOITuhBCMDg4CJ1Ox3XXTU1NiVouCzRrpGG+0+nEkSNHQob5k2YnTo6akZ+uRKpCCpvbh7WFi58tDra/n5mZ4ZxnPR4PUlJSkJmZmbA+P9RDgyUEMw4vdFYX6Fent0W3IAlRR2cY5jbM+sh9gczdPHOuMq65/z7KMEwfgGrMGlOEhShE59fGrVYr2traOJlo4OoQD9HDCWACa98GmxsdExasyklBhloGrdmFDJUcqQoZ1HMr+fj4OAYHB4MOWvR4PFFfl9frRWtrK1QqFdeiCiysYIYf5k9PT6OhoQFGoxE6nW5emL+/3QDCAAN6B66qL4RMykCT4t9LsBAreiQEDpgYGhqCw+GIa38fiGBZd0IInn5vCCdGzVidl4qm0ll75/OqcqI6ZqLlNYZhvgTg3wB8jhBi572eB8BACPExDFOB2RbV/miOKTjR+Qm30dFRjI+PzyMQH7EQPRoBDJ/ohBA8/o8eTM44kaaS4YEvVmPIYEdlbirUitnznjp1Ch6PJ6jKLZZ+dKvVipaWFr+IgGIxBTCBYb7FZsexvgm0DPdgetIGKxTISU+FWkqQkZp4UnQhQB136EM+mD6fEj9SojdY1t3hYXF8xIzCDCV6dDb86qp1yFRH3+sey4p+4403ArN+b/wW1QcBKAH8c+6hRcto5wH4KcMwHgAsgLsIIYagBw6AYEQPtHjq6OiAUqmMmGmOlujRCmD4x2MJYLJ7oFZI4fD4kKKQ4pzK2aey3W7nbJfLysqCrgLRhu7U3TWU7n0pSWD7jB4cnWYAZODCLeVgvG7AaUZvVwdYlvXzoqMPusVe0QMRGG4Hbl2C2U3T9xT4MOev6Ca7B89+OAyWEGwsycDxUTO2lGchQxUbTWIh+s6dO7Fz585AsUfQFlVCyG4Au2O6mDkIRnRKCpPJhM7OTlRVVXE+YOEQDdFj0arzV3SphMG3zl+Ftzp12FyeDU3qbNKFZqtra2u5pGCkYwUDy7Lo6emB1Wqdl3vgYykRxesjYDDrZiCXSlFbkg9gthZM+7mnp6fR29sLpVLJDSdYSoQPl4zj7+9Xrlw5b3/PMAyXzc/IyPBb0d/tNaB93AqGAb5Yk4s/3LAeCmnsDrFWqzWmRqyFgKChe39/P6anp9HY2Bh14304osejVQ8kZ92KTNStmLXapQIYs9kc1YimcCur2+3GyZMnkZ2djcbGxohZ9FiSemKu6BtWZIAQQCplsKbAf9UJ7Od2OBzQ6XRwOp04dOgQJ9oR0nI6HsSSdQ82QNJoNGJychLd3d1wu91Qq9UoKCjAikwFJBIADFCarYZSFl+Sz+PxLOrnEwyCEj07Oxvl5eUxZUFDET1erXoo4wmXy4WWlhZkZ2ejqakp6t72YMeiUtZAY8pQiCd0FwsKmQTN5VlR/S4lgNFoxIYNG7iSF3XHpWF+NIMYhHxwJdIcE6hoO3bsGKRSKQYGBuCz2XBDlRrpGRnYWBy/fkIoTb+QEJToOTk5MZejghE9kRFIwcprBoMBnZ2dWLNmDbdaRXuswBt0ZGQEo6OjMTXfBBKdZVl0dHTAaDRyZAnUfy+l7jVmbsABv+TFD/P7+vogl8u5lTPYhFUhQ3+hibRixQrIZDIQQrhW1c7Ozoj7+1DXthSx6G2q/C8/0bZSYH7WfXBwEFqtNqhjbDTXRo/l8/nQ2dkJlmVj7objE93pdOLkyZPIz89HRUUFt0pS/bdGo4FKpVoyN0yo6wgM851Op9+E1cAwfyE6zhI9VmCrKn+cFL9jTaPRhPSgo8dZKvkMCkGJnsibE2paKSUVHe6gUqni7oqjDw2Hw4GTJ0+GzdBHc000E7x27VpkZ2dz3Wu07dXtdnONPmazGT6fDzk5OYu6J46WoCqVCsXFxZwfe2CYn5GRwYmnEjVjFJrood5fpP091STw58THuq0I0b2mwazstRyzDrDXE0KMzOyBfwfgUgB2ALcRQo5Fc55FX9EBcJ1NQkwrZRgGPp8Phw8fRkVFRdwGFsAs0Z1OJ44dOxZXHzr/mgwGA0ZGRrhEZbAtjkKhQFFREdLS0jAyMoKSkhI/sixUI0uiCBbm6/V66HQ6HDt2LGKYHwlCWzRHe6zA/X3gnHiVSoWDBw/G1B8RonvtAQAHCCG/YBjmgbl//xDAJZgVyawGsAWz01y2RHOeRSU6DdWdTifOPvtsQZwzR0dH4XA4cPbZZyekTiKEYHR0FBaLBeecc07cHXYsy2JiYiLmkD8YWfiNLCqVilvtY7EWjhVChNy0VzstLQ319fURw/xorkmoB10i7y2wfq/VajE5OYm+vj40NTXhK1/5Cu6///6wxwjWvYbZQYrnz/33XwAcxCzRtwL465wk9mOGYbIY/0EPIbFooTs/VKcNDYnA5/Oho6MDhBCkpqYmRHIqZZXJZMjKyoqb5C6XCydOnIBSqYxpGGKwLD1f4Ua7vfR6Pbq7u+FyufySekthFHEg+KtwuDDf5/P5RS7B3osYFs2JgmEYFBQUYPv27ZiYmMALL7yAsbGIk5JCoYBH3kkAVJCyAsAI7/fogMWFJXq0CMyqf/LJJzGPUeLDZrOhpaUFJSUlKCkpwUcfhZ0gGxZUyrpq1SpkZ2ejvb09ruOYTCa0t7dj7dq1cDqdMWnmI4Hf7VVaWsoJlejgAplMBo1GI4jQRUjn1lDqw1iz+UuR6BTU012lUqGysjLh4xFCCMMwCWdmF5ToobLq8cxLo5iamkJvb2/EGdTRIFDKShtzYsXo6ChGRka4Etz4+Pi8FTqRNtVA0Gw9TRpRtxO3252w0EVsogcimmy+0+mE1+uNa9Za4DUJDYGGN0zRkJxhmCIA2rnX4xqwCCxg6B4uqx5PBxvLsuju7obNZgs6wSXSjeX1sTg0aAQhwKaVmRjs74PNZvOTssY6ZJFaQdOpMvQ9LnRTC3U7GR0d5Wav6fV6LqlHSc8vEXl8LLwsgVouTtgf7wMjWJhvNBrR2dkZVZgfDmLMXRPIGPI1AF8F8Iu5/9/De/0ehmFexGwSbiaa/TmwQCt6JAFMrER3Op1oaWlBbm4u1qxZM+/LomWxcF/8JwNGvHxsDD6WRWdXFy5YW4CNGzf6HSsW6arL5cLJkyeRl5c3zwp6MZta+KHxqlWr5pWI1Go1lGmZOKlnwDJSnF2R7TdhdaFX9HCg70WhUPj5zEcr2gmEGHPXaNQRLUJ0r/0CwEsMw3wdwBCA6+d+fR9mS2u9mC2v3R7teUQlerQCmFiITp1b1q5dyzlyBjteqC9x2uqCx0fAztXaZ2ZmkFdZFnQ/Fe3ARiqJDaW8W8w21UDwS0RkblpJ+9AUBkd1kBEWH9v0SKktjil5GA2EboqhvnqRwvxw7aqR9vpuL4v/PTSKQYMD1zcWYX1xRsTrinVKS4juNQD4QuALc9n2b0V9cB5EC91jEcBEQ3RCCPr7+6HX6yOq3KgMNjCc75+24amDA/CyLC4oU6Au3Y7yump8vqY46HGiuTHHxsYwPDwcVhIbiujhElQL8WA4NWVF+4QV5RoN6lanweryoj5fyvV204qI1+tNmKgLkUALDPOtViv0er3fOCkqZ6X3XLj7csjgQMekFRlqOfZ3TEdF9KU4vAEQaUWPVaseiehutxutra1IS0vzc24JhVAr8ZjRAafHB4fNjN5JBb7z5bPjXrXolFeXy4Xm5uawicRQxF1MmaTHx+LIsBmaFDk6p6y4tqEQKrl07ppmqzlOpxPj4+PQ6/U4dOgQ50Wn0WhiToQtdJsrX87KHxdNH2IymQxpaWlBKxMmuwcHe/RIkUuQoZTB4vCguSy6RK/dbk9Y9CUGBCU6IQTDw8Mxa9XDEZ2GxdH2twP+RDc7PPjje4OYcXhwZV0eUjwmZGam4ivn18ZNctqimpOTg7Vr10a8gYMR3ev1wmAwBA2RxVzRHR4fWkbNUMkkKMpQYGLGhfx0BZQcyT8FFeV4PB5UV1dzXnSjo7NmidnZ2cjJyUFGRkbEz2Cx+9kDBzDQh5jdbsehQ4c411mNRoPdJ6bQOm4BCHD7WaUoyFCiID26aoXNZltyI5MBgYlO7Zxj1aoHIzp/fx/rmCb+8donLOiftoFhvXj1w3Y8dEVDWLOJSIi1RRWYT1yHw4ETJ04gJSWFC5E1Gg1ycnJEn8rSMmZB24QFPgJ8vjoHTWVZSFNKIQlxTkpQvhcdP6k3Pj6OU6dOITU1lSNKsG3VYhM9ENRnzufzoaqqClarFYMTOhzubsX4lAtmmwwqpQIpcgaFGdELppbigEVAYKLL5XKsWbMm5r8LJDqdqCqRSOJqcOGv6CVZKvhcDticLmz7/LqESD4+Po6hoSE0NDTE9NTmE522QNbU1HCkpkkkvqe5y+WC1+tNaHRRMKjlErAsIGEAtVyK7JT4atGBST2bzcZNIvV4PPP2w0uN6MCneQOGYcDKVHihww2rS4W64lxsTAUkbhuMQ504Pv5pNj8tLS3s+1g2e/R4wk6pVMopx6xWK1pbW1FaWhp2wkU4UKJ7vV5MD3XhG42ZKFm5CkVxDiag+3Gn0xlxPx4M9DOhEUpTUxOUSiXnZ89PItFRVSaTCSdOnOBCzngbQAJRW5SODJUMcqkEK7Iit+1GQ1CGYZCWloa0tDSUlZXN2w/L5XIolUruc0hUqScU+H5xRrsHVpcXaUopRk0u3HnupwsWFSANDw9zRKbED8zmL8UBi8AS6V6TSqXwer2YmJjAwMAA1q9fj4yMyBnOcMez2Wzo6upKuCOOZVkcPXoUGo0mqv14MBBCYDAYOF97usIFA3U4TUlJQUNDA1wuF/R6PQYHB7nhDHTUcjyrvVTCoDwn+m1QPMQKth8eHBzEzMwMDh8+zCX1srOzFzWpxy/BlmnUOLdKg26tDVfW++eCqACpqKjIz5wiMJufmZkZd+jOMMwa+E9kqQDwMIAsAN8AoJt7/SFCyL5Yj78kiM4wDHQ6XUSTxWhhs9kwPj6OxsbGhEz6zGYz7HY71qxZE/V+PBButxtdXV2QSqWoq6uL6iblR0VKpdJvtTebzdDr9ZxRBSWUEKt9uOtJBHQ/rFarUVpayiX1RkZm+zNofiI9PT0qkYsYphMShsENTcHLrHwEM6eg0cs999yDjo4O/PnPf8ZVV12FDRs2RH2thJAuAA1z55BiVtr6CmZFMb8hhPw6rjc5h0UP3R0OB7q7uyGVStHQ0JBwrba7uxtWqxUVFRUJkZwOdVCr1THZT/FBp7+sWLECZrM5YcLwZ3YDnxpVUIEIf7VP9GFJIbQyLvA9eDweLpNvsViQmprKbVVCiVyEeqAJYYLBj1527tyJ888/H6WlpXjiiSfwzDPPxNv5+AXMTk0dEuq9LuqKPj09ja6uLpSXl3MzweMFlaDm5OSgpKQk7mPRhwWtHhw+fDium31qagp9fX3YsGEDCJkd5xwton1YUqMKGlLS1X5kZAQMw8DlcsFisURMIC0EQn2Gcrncb9iizWbzE7lQZRvfZ36h5q7FA5/Ph69//evYvn17IofZBmAn79/3MAxzK2ZHL91PCDHGesBFITohBH19fTAajdi0aRN8Ph90Ol3kPwwB2hJKJaijo6NxdZ253W60tLQgKyuLiy5oYi+GEAz9/f0wGo3cNsRqtYqudOOXv4DZ93LkyBEugZSens6tlLGs9gupdecn9WhYbDQa/Xzm09PTwbKsINe1FNtdGYZRALgCs9NagFkXmZ9h1or/ZwD+E7MTVmOCKKF7OFAyZWRkYNOmTdzKE8+gRZrJpvtx6rQS68w04NMwu6qqirMKoseKlqQ+nw+tra1QKpVobGxclNlrFAqFAnK5HLW1tVzXl16v58QuseyLhUA8IbdUKp3nMz8xMQG73c4l9RJJTAabu5YIBHqYXwLgGCFkau6YU/QHDMM8A2BvPAdd0BWdrryrV6/2I1M8bao+nw/t7e1gGIbLZFNE24xCQbP9wWbERdvBRkUwwcqCi93UEqyDjZLeYrFw/eo5OTkxt/tGC0JIwvthtVqNvLw8OJ1OrF27dl5ikpa8on14idG9RsVFCeBG8ML2AKuoqwC0xXPQBSE6lcZOTEwEVbnFSnS73Y6TJ09yjjKBH2yoIQ7Brovf0x5sVYhmRafurqEMJBezTTUY5HI5CgsLUVhY6Lfat7S0APBf7YW6DiEfGBKJJGhikp/UC1frphB6RU/0eAzDpAK4CMCdvJcfZximAbOh+2DAz6KG6ETnd7EFrrwUsdzYOp0O3d3dYeemhZqRzofH48HJkyeRlZU1rw898NrCPTRGR0cxOjoatqNusVf0cAi22vMJI5PJoFAo4Ha7E0pcCUX0UPtqhULh9/Cite729nZuqkxOTo6fg67QK3qiOndCiA1ATsBrtyR6XYDIe3S67w02SjhW0CSXwWDApk2bwpYtIoXu9Lo0RSthkaXCaPdwAxijPVZg91q4G2aprejhEJgFHxoagslk8nOnibaRhQ+xic5HYK2b+tDxHXQ1Gg08Ho+gK7pANlKiQLQVndahQ40SjgUejwetra1ITU1FU1NT3G2qwKe+cOtq1+NPh7UwO8zIVMtx7wUVkEnnHzcY6YToXtNqteju7kZ6ejpyc3Oh0WgE17UnCoZhOLHLypUr/RpZurq6kJKSwtWQI632C0n0QAQaVNjtdhgMBlgsFq7KEmwsVqxYqvJXQASi0ySZx+MJue+NBXT1jWUYQ7A9OiEEPT09MJvNqF7fAKVSAYfbB5VcAofHBzbEAhr40KAusYHZ+XDgE53MjYmanp728zgfHh72U7otpVCfEjRYIwvfi47fthpIRqGELkI8MKiDrlarxfr167mGHH5SLycnB70mFh8PmNBYloEt5ZGHdyyrFX1mZgapqalYuXJlzF9I4JcYLhseDoF7dI/Hw5X0RqRFeOWdQVTkpOD6xmK0jluwsSwTihAjcvlE12q16O3tjfl6KNFZluUqBQ0NDfB6vZBKpUhNTUVpaSnXo0517V1dXVz5iL81cHtZ/Pd7QzDZPbjn/HLkhNh2CIFwLjj8mjcdMEG96FJSUjjCKJVKwYQuQktg6XAJmkSlDSz9g0N47qgJmSkKDGlNqM5VITstfFPUsiI6faLHCr7PG79bLJ6ogE9OugJXVFSgoKAAf3mtE4XpSvRN23HVxmJc0xh+W0GTcdTGatOmTTEnpRhmdkzUkSNHkJ+fjxUrVgCYXSEJISCEcHbXdMW02WzIy8vjvNrlcjm32u/tNGHHkTH4WMDpZfHY1rUxXY8YCBwwQVfJjo4OzsVFoVBwCrd4IfbcNaVSCbMkHchOQW1FGkb0NqRJPOg51QEJSNixWMsqdI8XdBX2er04efIkcnNz4+4Wow8LKkNNK67Eaz0OrDFP47zVOXi3R4+64nRkqaNTiPX19SE9PT2q/EAw2Gw2WK1WNDQ0QKPR+K1u9P3RFdvn86G7uxsZGRl+CSWPx4OxKR1OdHRjdNQOwhIwAFQhIhGhEE+oHNi26vV60d7eDpPJhMnJSajVak6lF+uEW6HVbIHv7dSkFX89NAqWBS6o1uCLNfkozFAiRSGNOBYrEdMJhmEGAVgA+AB4CSGbmBDDFuM5/oIr40JBJpPBYDCgr68vrMNrtNdgtVoxOjqK5uZm/Oqtfri9LPqmrfju56twwZo8yIMk3gLhdDoxNTWFgoICrFu3Lq5roeVAmtQCEPJGpaXI9PR0VFRUcOE+y7IYmfHg9t2jcPsIHrxoFe7S2DBltOKcbCNaWlq41T5W4iwEZDIZVCoVCgsLkZGRAbvdzrn50omygaWvUBBS6x54XJvbB7t7Nl8jYQAvS1CR+6nmI9xYrPfffx9vv/02ysrKEjGfuIAQMs37d6hhizFjSazo9EPr6+vzk7LGA7ofJ4SgsbERDMOgIF2JU1NWpCikSFXKoiI5VfHRJ3asoKUprVaLxsZGtLS04NixY1z2N7CtlHrVl5aWcv3zVGsPAB8PW+D0smBZgldbtPjTTeu533E6ndxQA9oIkpubK0hCT2itO8MwSE1NRWpqKmdSEWyVDPXQovtqIcB/X39v1+LdXgNKs1U4f7UGHh/B+atDLzYM4z8Wq7q6GqOjo+jp6cEFF1yAn/zkJ7j88ssTvcRQwxZjxqITnWbpfT4f1q9fnxDJ+ftxl8vFfZFf2VyKvmkbCtKVSFdFfsvUwrmxsZGbhBoLWJblBj5u3LgRANDU1AS3280NG7Db7RwhpVIpTp06hXXr/K2uCCF46dg4tBYXzl+dgxS5FE4vi69sLoFCoYDP5wPLslAqlX5iERoi2+12tLW1RV0CCwaxy2J8PTt/leRbUuXk5HB7eyFDd/6D8KNBE/LSFBgxOnFNQxGKMmOLjGg78znnnIMbb7wxnocsAfAPZnbO2h8JIf8PoYctxoxFDd35UlaamIoXNCNO6/b9/f3cz9QKaVSe3LRFlW8ZFetYJlpjz83N5TTvdCVTKpXcmF2WZWE0GjE8PMy5wVqtVqhUKm4le6NNi1/+oxdeH0G/zo6D3zsbbi/hHlZ8hRdN6NEyV3Z2NmZmZlBWVgaj0egneMnNzY2pmWUhu9f4qyR/Ektvby9UKpVgraWB3+l5VRocODWNqrxU5KbFd3x+1j2Oz+yzhJAxhmHyAfyTYZhTAdeb0LDFRVvR6d6VDkfs7u6Ou4Otr68PJpMprow4BZXEZmdn+415irapBfCvsdOkW6gmB4ZhYDabQQjBeeedx6327e3t8Hq90Gg00M8wIARg5/aPSpkUyiDfGCU8P6E3NDQElUoFtVoNtVrN9awbjUa/ZpZIrasL2aYaiEBLKlpyHBsbw+joqN9qH8/cNX5kcNHaPJxXlQOFNP6mlESy7oSQsbn/1zIM8wqAzQg9bDFmiEL0cBJOfi96c3MzR8x4Oti8Xi9aWlo4xVy8XxAlaGVl5Tzv+Gg74aiJxvr167mmnVAhJsuy6OzshEQiQUNDAyQSCeRyOac/oPX0BqcW569gYPUpcc+WbHg8noi95FQY5PP50NDQwL1GV3tKbIZhYLPZOOIzDMOt9mLYUgkldElLS0Nubi4yMjL8DCgVCoWfZXYkBNO5KxOsYMRL9LlmFgkhxDL33xcD+ClCD1uMGQu6otNEWVpa2rxSVaxEt9lsOHnyZMLmjzSy2LBhQ1CpbjQr+tDQECYnJ9HY2AiZTBa2VZF+Bnl5eSgtLQ36e/x6+hPrZ7vLpqenceLECQDg9rSBzjG0H542qPB/Rm9qlmU50tOkWElJCVc6orZUmZmZyMnJCTraKh4IvdcPXO3p3r63txdOpzPiai+G6UQCgpkCAK/MfT4yADsIIfsZhjmM4MMWY8aCET2SlDUWotMMbV1dXdxusXwpKj+yCEQ4Ews6Jtnr9aKxsRFA+H5km82G1tZWVFZWxjT8gXaXVVRUcCH+wMAAR8i8vDykpqaira0NJSUlYRuIaIsnvX6+WIeWjui2girdZDIZ3G53QgMmxE7qqdVqrm2ZZVmYTCbo9XrObpo+FNRqNSdgWuxJqhSEkH4A9UFe1yPIsMV4sCChO21wCScdlUqlcLlcYY/L72ALR056DaFuCprpl8lkEUUwoZJxdE+v0WhQWlrKnTPUzWwwGLj22kSafBQKhZ8rrMlkwsTEBCYnJ5GWlgav1wuHwxFV9YK/t6dRC13tqViHYRjI5XJIJBL09fVFtVoGg1AraDSaeb4JBTBbuuSv9tROW2gs1SktgMgreixS1kgrutfrRWtrK9RqdUwdbIG/53Q6cfLkSRQXF3MEDYdgoTvdNlRWVnINKOFIPjY2hvHxcWzcuDFeV9CgoFNGzGYztmzZAqlUiunpaXR2dsLtdnN77mhkp5ztMW+1t9lsmJ6eRnV1NVJTU7kmHmrX3NfXB6VS6bdahoLQxhOxQKVS+VU7ZmZmMDExAaPRiOPHj3PXn+g4rGUpgaUCkLy8vKikrOGIbrPZ0NLSgpUrV0bd1x6sg42KYNatWxe1CCYwGUdrvDTpFu7GI4Sgt7cXdrsdjY2NgoeKExMTGBkZwcaNG7mSXGlpKVeaMhgMmJqaQldXF1JTU7m9fTSVCavVivb2dtTW1iItLY1T6fl8Pj+jCpfLBaPRiK6urrAqt8VsU+VDIpFwCkW5XI7S0lK/cViJ+NB5vV7BbLaFhihEp3XbWKSsoYhOk2Wx7scDO9jo3LRYBzbyQ3e+HZZcLg+7ivt8PrS1tSElJQUbNmwQNItN8wsmk4lLAAZCKpX6yTWtViump6dx8uRJALPNR6Hq6Xq9Hj09Paivr/f7rKRSKeRyOUd42qhSUFDArfYzMzNcDoVq2vlRT6IQcgsgkUjmjcOamZnhWlalUimXyY9UiVhKbcXBIArRXS5XzFLWQKLHsh8PBroSU184u90e99w0n8/HhcPUdiocyanHfKTEWDygCUAAqK+vj+qm5zuurFq1ihv8MDQ0BKvVioyMDOTl5UGj0UCn02FkZASNjY1hE5ShxDp8Hzcqze3o6IDVasXAwABXGouX9EL1tQfzd6OrPb9lNdiAjHAmIUKXJYWCKEQvKiqKuSbOJzrdj6tUqrg7xiQSCdxuN06dOoXMzMy4p8CwLAutVouSkhKsXr064n7cYrGgra0Na9asiUsjHw70c8nKykJ5eXncN1Xg4IeZmRnodDqcOnUKLMty3XLRPFwDxTr81V6lUqGoqAiFhYU4fvw40tLSuDHLaWlp3Gofq8+8UCt6pK1UqHFYfJMQOmEViI/kIyMjKCsrewezJTYC4P8RQn7HMMwjEGDmGoVoWfdYQYlOZbFlZWVc33Y8YFkWbW1tWL16ddTONIGw2+3o7OxESkoKysvLI5Jcp9Nx01mENiCgUQK/6UUIMMzs4AetVguNRoOKigquQuByuTg9fnZ2dlQEC7ba0yk81K6JYRjOzok6z1LSR5oqI3ToHi1CjcMaGhrCzMwM/vjHPwKYzQPFMpp7LjK4nxByjGGYdABHGYb559yPf0MSnLnGnUeIgwgBqVQKh8OB48ePc7LYeKHT6aDX67FmzZq4SU7nmK9evRq9vb3Q6XTIyckJenNQO+vp6Wk0NTUJnpCho6TFiBJoA45CoUBtbS2nNy8pKeE6y2ieJCUlhUvoRVM9kEgk0Gq1GBwcRGNjI+RyORfiU007/zx0qky4EFmovX6idXR+VEQ1+Q8//DAuvfRSNDc343e/+11Ux5mLqo4BwJwyrhNA/CtcCCwJolOiOBwOnHfeeXGXoPitoUVFRXH3ZlMLZ5p0o2Wr/v5+qFQq5ObmIi8vD0qlktsz0041odVWdHWtq6vzK90c7J6G1uLGZXX5SFXE9zVSCXFOTg5Wrlw57+eBnWW03Nba2so1yOTl5YXcc4+Pj3NlRfrw46/29H/Ap8lBhmFgsVg44vMVcELWvoWcuyaVSlFfX4/Vq1fjjTfeiHlKEAXDMOUANgL4BMA5EGDmGsWih+7UbEGpVCIlJSVuklM/NolEgk2bNqG/vz/m9lJCCFf3p73sNOSkK6nNZoNOp0Nrayt8Ph88Hg/y8vKwevVqwUnOL5/xP5cP+w340eun4GMJTo7N4NErYjfFcLvdOHHiBMrKyqKKeviuMeXl5Zz/+8jICNcgk5uby+25h4eHodfrsXHjxqArZ7AQnxI/LS2N61enzrO0/OVyuTA9PT3PRy9WCD28ge/pHk9ExzBMGoDdAL5DCDEzDCPIzDWKRV3RA/fjH374YVzHoSKYoqIilJWVAYhuiAMf1MIqIyMD69evD7kfp/rw/Px8TifgcrnwySefICsri8teJ3ITEUIwMDCAmZmZoOUzs9M729XGAiZ77KuH3W5HS0sLVq9eHbeTT6D/u9lsxvT0NIaHh+FyuSCTyVBbWxv1vh7wT+jRpB5fmgsAhw8fhslkwsDAABQKRVRinWAQWuueyPAGhmHkmCX5C4SQvwHCzVyjWDSi87u9EtmPz8zMBK3ZxzJ/jT5wVq5cyVkZh0u6mUwmdHZ2ora2lqvtUzmqTqdDT08PUlJSkJeXF7VAhYLf2cYvnzk8Pty/qx3903Y89KUqXNdYjDGTA/deUBH1sYFZVRsVwsTbJxAImtDLyMiAx+OB1+tFVlYWBgYG/Aw2ol2F6Wovk8n8Vnuz2QyZTIaVK1eivLycE+vwE4d8k4pwEHpKS7z2UXP19+cAdBJCnqCvMwLNXKNYcKJTsYdOp4s4cSUSwolgoiU6rfPW1tYiNTU1YvkmmBqNno+G+HQ/q9PpOIEK3deHE17QPbNGo5lnl/1BnwEnRmfgYwl++84A/nZHc8T3FohQQhghQLdOarWa6+enDSaBo4/pZxFNDoV+FzabjfMvkEqlIcU6fJMKuu8Pdo+JEbrHQ/QPPvgAAG4B0MowzIm5lx8CcCMjwMw1igXdo1O1mFwux6ZNm0JmsCPt8akIxmazhRTBSKXSiEmRsbExjIyMoKGhAQqFIuwqTgU8FoslpBqNgr+fpQKVQAupvLw8v5IVlQwH7pnHZ5wYMzqwUqOGRMKAAYOGFbGvxPQBFU4IEy98Pp/fA4oP/lAKAFxCj2+wkZeXh8zMzLAR1KlTp+Y9oMKJdag/e0dHB3ce/nCJpTJ37bOf/SwIIcHeeNw182BYsBU93Fhhimi+AP5eOtxwxHArOl8tx0+6hXtAtbe3Q6VSob6+PubyTmDHGb9klZqairS0NExOTmLdunV+01hHjA5se+4ofCzBBdU52Pm1JkzMONFYFttWZ2hoCHq9PuIDKh7QLj76/iKB5jj4BhtjY2Po7OzkTCVyc3O5hBatOjQ0NMyLAEKJdaiPHi1/0W3V5OQkp/t3OBxxORqFwlLuXAMWiOi0ESTcBFTgU9FMKKLTrrFoxjOFIjoNj9PS0lBXVxdxP+5yudDS0oKioqKQD6hYEDh2aXR0FP39/VAqlejr6/ML8ft0NvhYAq+PxfERM0qz1SjNjj7pRN1m3G4352QjJKiIp7y8POrxVHzwDTYImW+woVarYTab0djYGHWYH0yaC4AT60gkEtjtdnR0dKC7u5urqtAx0fHW6G02GzfbbSlC1NCd1rWnpqai2o+H62Cjybtom1uCda/RqKKsrIzLFocjudVq5dR1ifjMhwLtIz/rrLOgUCi40lFPTw+cTidyMrJQW6BGv8GF710YW9KN7plVKhUnhBESDocDJ0+eRHV1tSAinkCDjfHxcfT39yMtLQ3Hjx/nDDY0Gk3UCT0g+GqvUqkgkUi4qkCgjx5NHMZSJrPZbEG1CEsFoq3oNNyVSqVobm6OajUJRnS+CCaW5F1geY22qNbU1HBtl+GuiSZ11q9fL3hIFrjfpzcj3yWWtpl+fzOByeRBmncKk5NsVNpwGrXk5uZy5UYhQZV6NTU1CVVMQmFiYgJjY2PYsmUL1y1H3WD7+vqgUCi4yCfashpd7Qkh6OjogEajgUKh4Hz0cnJywMwN/gg29DJS99qyDN0JITh69CiKioqiMnegCCQ6lWcCCJm8CwV+6E6z8/X19VAqlWFXcWC20WBqakqUxBUtn1E1VajrCGwztVgs0Ol0nFqM3uiB2XMaTkcrhIkVMzMz6OjomKfUEwpjY2OYnJzExo0buXxCoGOMw+GIy2CDklypVKKyspIzFeGLdfjDJfg+ena7PWyv+rIkOsMwfrLHaMEPt10uF06cOIHCwkKUlZXFZRXs8/nQ09PDrZzUkSVcZr2rq4vzgBN6T0vnvIeSnIYCP6ytrKyE0+nktjIul4srIykUCrS2tgoWTgeClucaGhoSGrQRClRN19DQEDY8V6vVMRtsEEK48l9FRQV3D4QL8fkSYGC2M5FOu+X70KWkpAjiLsMwzJcA/A6AFMCzhJBfJHRA/rEjNMzH3U3vdrtjbsbv7u5GdnY2FApFzMYVgTCbzTh69CiKi4tRWVnJKaHC1bBbW1uRmZk5z0FVCFD1Xnl5+TxL6UTg8/mg1+sxPj4OvV4PjUaD4uJi5OTkCJphp80ptBQpNAYGBmA2m1FXVxf3A5ZvsDE9PTvCjJJxeHgYKSkpqKysjPp4geU7CoZh4Ha7YTQaodfr8cc//hGnTp3CN77xDdxyyy3x9FgwDMNIAXQDuAjAKIDDAG4khHTEerCgJxCL6B6PJ2ateV9fH9xuN0wmU0KiDqfTiWPHjoEQgi1btkRMujkcDs6qSoxwl/aoB45cEgp0pd2wYQM8Hg/XvSeXy2PeywbD2NgYJiYmUF9fL3hnHvX5dzqdqKmpETSKop9FX18fWJblPotwxhHhwO+1pw8BYHZR27ZtG1avXo3jx4/j4MGDsRqAMgzDnAXgEULIF+deeBAACCGPxXyhQbAkuteA2S/cYDDA4/HENROdgkpiq6qq0NXVBZ1OF/aLpXtOsUhIk3pi9KgDs4mr0dFRv3xCZmYmqqqq/PayHo+HC/HDiVMCMTg4CKPRGLI5JRFQPYPP5xOlMkC7DktKSlBeXs4ZbAwMDEAmk/kNvIwGocp3Pp8PHR0deOGFFxIpwa4AMML79yiALfEeLBBLgug0S8wwDFasWBE3yScmJjA4OMgl3Wpra6HVatHf3w+1Wj1Pez41NcWFo2LsOan7qxhJPeBTEoYynuTvZb1eL/R6PSdOofZROTk5Qf+WrrQOhyNqy6pYQAjBqVOnIJFIsG7dOsFJTo1HqH0WgHk2V9PT0wkZbACzq/ntt9+On/70pwkZpYgN0UJ3r9cblfKINpRQBxe3243y8vKYzkVvypmZGaxfv35e0o2vPdfpdNykDzq2SGi1GL0em83GabOFPn53dzc8Hk9c4S7fPspgMEAul3MZfpVKBUII11jDn0MnFGg1RaVScdlvoY9P8y3R3EvU+GJ6ehpGozFqgw2Xy4WvfOUruOKKK3DXXXcl8j5ED90Xleh86+TMzExMTU3BarXGlDChY4hUKhWqqqoi7sfpTeDxeLgpLHTvlogyin/8jo4OyOVyVFdXi3ITUyFMVVWVIMe32+2Ynp6GTqfjvresrCysXbtW8JU82Eor9PGpr148Aha+wcb09HRIgw23241bbrkFF154Ie69995EvweGYRgZZpNxXwAwhtlk3FcIIe2JHJg7wWIQnRCCkZERTExMoKGhgXtqTk9PcxZQ0cDpdOLEiRNYsWIFZ3QYjuRutxstLS3Iz8/nhCRer5e7ya1Wa9CGk2jBn6smhlBFbCGMz+fD8ePHuayxxWKJWZEW6fjU0UaM62dZFi0tLcjOzhZMpUYNNnQ6HSwWC9xuN3p6evD222/jnHPOwfe//30hHrYMADAMcymA32K2vPYnQsijiR6YO4FYRPf5fPB6vfNepyseIWSeMQGd9bVuXWTHFH4femZmZkSS07lnVVVVITXJ/IYTo9GItLQ05OfnR1Wqopn7VatWxaX7jgS+ECa/oAA7Do+iV2fD7WeVYaUm8ZZTj8fDPTRpcwr1OachvlKp5EL8WNuLfT4fTpw4gYKCAkF6BgJBSa7RaER5iACf6izuv/9+9PX1oaqqCg8++CAuuuiiRA8tukf0gibjqH1Rfn7+vH5rIPpBi5OTk+jv78eGDRu41Sfc6hvt3LPAhhOLxcLVj+VyOfLz84Pe5NTMQSxJKH1IUSHM4UEj/vzRCHw+gnGTC//vpnnz+WICFSdVVFT4DX8M9DkP9IyjW55Izq20w41GXkKDRgq5ubkxKTFjBcuyeOKJJ3Duuefi7bffxuTkZNz+cAuNBSO62WzmnExDraiR7J+oRpyfaY4kZ4137hlfjVZVVQW73c55xRFCkJubi/z8fNjtdvT19Yli5gB8Wv6jmnuX14dUpQwMZsOtDFViXyG1lYpGTcdvMfV4PH5TXamNVqCLDH24r1y5UlChEMVCkvy+++7DihUr8Mgjj4BhGFEeWmJBtNCdZVnuaUdX4Pr6+rA1S6fTiY6ODm4EMR9804rVq1fPXnwEOSvtAqutrRU0802NJIaGhuBwOFBcXIzCwsKY6tPRgNbg6+vrAZkC39rZgp65cH1dYRqGDQ58qTYfmer4RCy0OSVRWym+jZbBYOCy1hkZGejo6JgXKQgFn8+HkydPIj8/X5TtAAXLsrj//vuRkpKC//zP/xQ8QYnTPXSnQwbNZjOam5sjqqpChe40tCwqKkJxcXHE/Th9KKSmpqKurk7wzLdcLuccRZqammAymbj6tFDJq/HxcYyNjXE1+BMjMxg0OKCSS/DS0XHs//ZncFZsnat+oK4tQgh5gtloTUxMoKurC2q1GlarFWq1OmIHWCygJC8oKBC1fs2yLB588EHIZDKxSL4gEI3oXq8XJ06cQEpKCufiEgnBiM4P+bOysiKSnFoyiTH3DPi0PKRWq7mHCN88ga5sfX19nEgnLy8vaukobcul25NxsxvtvVrUFacjJ1UOndWNL65PzOCA35wSr/d9KDAMA4lEwjnapKSkRLTRihU0sUcf/GKBZVk88sgjcDqd+OMf/3jakhwQMXS32WzQarUxfRGEEHz00Uc4++yzAcwq1+iIo2jaS2lSbO3atX6WTEKBJpUKCgoi7gcDRTr8ttNQKjwqhPF6vVi3bh3MTh+ufeYI3F4fSrLV+POtDTDYPCjMUMa9Mk5NTWFoaEi05hQ64jrYdiCwqhHrKGdg4UhOCMGjjz6KsbEx/OlPfxJc9BSA0zd0T0lJifmL4CvZBgYGuFUhmqQblbqKlRSjjiqVlZVR7TcDDSKdTid0Oh2nOw8U6dBIISUlhRPamBxOuL2zjUFjJicUUgmKMuNfgUdHR7k+e6HVgMCnzTuhetUDqxr8Uc4Mw/j12Af7rqlfYHFxsaiJMEIIfvWrX2FwcBB//etfxSb5gkC0FZ3KWWPFBx98gPT0dEilUlRXVwOInHQbGhqCwWBAXV2dKIPoaeZbqPJZoEgnMzMTZrMZRUVFfkIPQgieeX8I7/bocdtZZbhoXfwJLToQoq6uTpQbl35G8e75qY2WTqeD0+nkQnxqJkG3gmKV6CgIIfj973+PY8eOYceOHaLcT0Eg+oq+pIjudrvx7rvvorq6mvMEjyRnpbPCxZBrArORwsDAADZs2CBK44vD4cCxY8egVqvhcrmQnp7ONZsIserShKjL5RK8DZTCaDSiq6sL9fX1gnxG/OGOJpMJKSkpnCebmIk3QgiefvppvPfee3jppZdE2dqEwOlLdGD2KR0tLBYLWltb4fV6cfbZZ0dMulG5KZWDCp1ZB2YtpbRaLTZs2CDKkz1QCENHG9F+coVCEbcSDQDXnEKjIzE+I71ez5UAhU7sAbMP/6NHjyI1NRVOpzOsjVYiIITgueeew5tvvom//e1vCQ0WiQOnN9GjdZnRarXo7e1FXV0d+vr64PF4UFBQgLy8vKBPVSryqKioEEVuSmvwLpcr6vlhsYIvhAml1qMiHZ1O5yfSiSY0ps0dtHlEDJLT3m6xEntUlktde4FP20t1Op2fjVZWVlZC7/Evf/kLXn31Vbz66quiRG4RcGYTnY5nmp6e5vaODMP43eASiQR5eXnIz8+HSqXiwsSamhrBZofxQd1rqe2QGAThC2GivamoSEer1c5aQefkhJxwQptf8vLyRFOLTU1NYXh4GA0NDaJEO5TkdB5eMFC/OJ1Oh5mZmbi3PTt27MDOnTvx+uuvi5LIjQJnLtH5Dq+0Wy1YqE6z1VqtFg6HA4QQrF+/XpTyGe1uKywsFE1pRYUw9fX1ca+C1CdOp9PBbDb7iXRYlsWJEydQUlIiWtKKzj0Xo5cfmCX58ePHYxoMEbjt4U9hDfcw3bVrF5577jm88cYbCZs7+nw+bNq0CStWrMDevXsxMDCAbdu2Qa/Xo6mpCf/zP//D+fffeuutOHr0KHJycnDo0KFVhJDBhE4eAaISPZRvHL+5paSkJOJ+nBo5WCwW5ObmYnp6Gi6Xi1vpIzVVRAO6HQjX3ZYI+EKYDRs2CJb55ot06OdSXFyMiooKUVbakZER6HQ61NfXi5K9pyRftWpVQrJZukDodDp4PJ6gM9727NmDp556Cnv37hWkmvLEE0/gyJEjMJvN2Lt3L66//npcffXV2LZtG+666y7U19fj7rvvxlNPPYWWlhY8/fTTePHFF3HjjTe+RAi5IeELCIMFJ7rVauUIRRNQ0cw9U6vVfkYLtESl1Wphs9mQk5OD/Pz8uPTmwcYgC4lAIYwYe37+6Ge6j41GpBMLBgcHYTKZsGHDBlHeA10AEiV5IOiMNxoBHThwAF6vFwcPHsSbb74pSHQ4OjqKr371q/j3f/93PPHEE3j99deRl5eHyclJyGQyfPTRR3jkkUfw5ptv4otf/CIeeeQRnHXWWfB6vZDL5XoAeSRW2+QYsKBtqnR2eF1dHWdZFO6GoXPPiouL55VVZDIZCgsLUVhYyO3VqN48KysL+fn5UcksaflMLN+4YEIYoUGFKvzEXkVFRVCRTjwREBUwWa1W0UleUVEheEQVOOPtxIkT+NOf/gSFQoGbbroJu3fvTvi7/853voPHH38cFosFwGw1Iisri9valJSUYGxsDMBsRyXNncz9fAZADoDphC4iDBaE6PyxSnQCRySlm8ViQXt7e1Ttk/yVi3ZSabVadHd3Iz09nTOPCAw1h4eHodPp0NjYKEqYS5Vc+fn5oiXFQo0UBgCVSsWZQ3o8Huj1eq6tNDs7G/n5+VFNN+nt7YXb7RalQQiYJfnx48dRVVUlyow7Pg4ePIgXX3wR77zzDvLz8zE6Opowyffu3Yv8/Hw0NTXh4MGDwlyowBCV6FTa2dnZCZZlsXHjRu71cDcMbQqpq6uLWWUV2ElFHVL6+/uRkpLCkb6/vx8ejwcbN24UZYWijjBi9WED4JpFomlOkcvlXARENed0ukmobDV1VAGAmpoaUUhOOxMXguTvvfceHn74YY6YAARJun7wwQd47bXXsG/fPjidTpjNZtx3330wmUzwer2QyWQYHR3lotIVK1ZgZGQEJSUl1IUpE4A+4QsJA1H36Ha7HceOHeNMASLtx4FPV9m6ujpBa7NUWz01NYWRkRHI5XIuqyt0DThQCCMGJicnMTIyklD2HpifraYindzcXG6goVAmlIGgJF+9erVonxPFRx99hO9///t4/fXXRe1dP3jwIH79619j7969uO6663DNNddwybgNGzbgm9/8Jp588km0trbyk3EvE0KuF+2iIDLRe3t7oVAouCaGSHJWmrASS6rpdru5pojs7GxotVrodDq/VtNE1V3RCGESBc18b9iwQfDylt1u5+yzpFIpSkpKohbpxAKXy4Xjx4+L+jCkOHLkCO69917s2bNH9NHGfKL39/dj27ZtMBgM2LhxI/73f/8XSqUSTqcTt9xyC44fPz5rDXb4cCUhpF/M6xKV6HSFoD3KoUAFHtnZ2SgvLxdl9aCrbLBZ5/xavc/n48p2sd7c8QhhYgEVGCU6oywcqMliVlYWiouLuRJVJJFOLKDuvWvWrBFFD8HHiRMncPfdd+OVV15BRUUCTh3i4vQWzGzfvh0tLS249NJLsXXr1qBZZ+qeKvTwQT5o+SyaVZYq0KampuB2u6POVAshhAmHhSjRUdeWYIq6YCKd/Px8aDSamK5lIUne1taGb3zjG3j55Ze5TsglitOb6MDsKrdnzx7s3r0bU1NTuOSSS3DllVdi3bp16OjogNlsFs09FfjUaIHvGBstoqnV01WW1pfFEJHQhCb1yxMj4qEVgmgMHagNtFarhdFohFqtRn5+PnJzc8NWLyjJ165dK8qcOz46Oztx++2348UXX0RNTY2o5xIApz/R+TCZTHjttdewe/dudHR0gGVZ/Pd//zfOPvtsUWZ70T51IfaytFav1WphNps511OdTgeWZUVdZdva2pCRkSHKZBPgU115aWlpzNNkqZOOVqsNK9Khxh0LQfLu7m7ceuuteOGFF1BXVyfquQTCmUV0it/97nfYv38/rrvuOuzbtw/d3d248MILsXXrVjQ1NSVMGFoW8vl8ohCQZVkYDAacOnUKXq+XC+9DDSyMF3SVFWvoASC8Go0vPfV6vcjJyUFGRgZ6e3tFjdwoBgYG8JWvfAXPP/88V849DXBmEr2/vx8rV67kSGG327Fv3z7s3r0bbW1tOP/887F161Zs2bIlZuLQWWx0BRQjzKW98FSrbzabuRWN1urz8vISiiIoAcvKykSZ2Q7MkvLkyZOi1bA9Hg8mJia4Mh3tI48k0okXw8PDuOGGG/Dss8+iublZ8OOLiDOT6OHgdDrxj3/8A7t27cKxY8dwzjnn4KqrrsLZZ58dkThUMssfKyQ0wglhaK2ekl6hUHCkjyVBR/eyYjXYAJ+G0mImxWijUE1NDdLS0ji9uclk4kQ6ubm5gkRBY2NjuO666/Dkk0/inHPOEeDqFxTLj+h8uN1uHDhwALt378ZHH32Ez3zmM7jyyitx7rnnziNOuPKZUKDniJYctCYdS62enkPMvSx1ahUzlOaTPLBRKJhIJ54HIsXk5CSuvfZa/OY3v8HnPvc5od7CQmJ5E50Pj8eDf/3rX3j55Zfx3nvvoampCVu3bsXnP/95HDlyBIQQNDQ0JNxTHAqJCmGiqdVTu2oxxTZ0OouY56APkmjPQR+I09PTIIRwybxodAxarRbXXHMNfvnLX+LCCy8U4vIXA0miB4PP58P777+PXbt2Yc+ePZBKpXjooYdw1VVXieIQQrXyQhlE8mv1LpcLubm5UKvVGB4eFs2uGvj0QSLEdJZQiJXkgXC73fNEOvn5+X6zySmmp6dxzTXX4Gc/+xm+9KUvJXTdIyMjuPXWWzE1NQWGYXDHHXfgvvvug8FgwA033IDBwUGUl5fjpZdeQnZ2NgghuO+++7Bv3z6kpKTg+eefDzpKLEokiR4Ozz//PF566SV897vfxd///ne89dZbqK6uxpVXXomLL75YkNWdDmkUyzLJ6/Wiv78fY2NjUCqVXAZf6Dlu/BFMYj1IKMlD+brHilAinaysLFgsFlx99dX40Y9+hC9/+csJn2tiYgITExNobGyExWJBU1MTXn31VTz//PPQaDR44IEH8Itf/AJGoxG//OUvsW/fPvzXf/0X9u3bh08++QT33XcfPvnkk3hPnyR6OExPT/v1/LIsi2PHjuHll1/Gm2++ifLyclxxxRW45JJLYt6LUiGMmF7owOwNNjo6ioaGBkgkknm1+mj76sOBjo0WYwQTBd0SCEXyQFCRztTUFG6//XZYLBZs3boVP/nJT0TJZWzduhX33HMP7rnnHhw8eBBFRUWYmJjA+eefj66uLtx55504//zzceONNwKYtUOjvxcHTt9JLQuBwIy0RCLBpk2bsGnTJjz22GNobW3Frl27cPnll6OgoABbt27FZZddFrGJgtbhWZYVzWgBmC0HTU9Pc9NoAMTVVx8OtJU11rHRsUBskgOfzmqXyWRIT0/HzTffDJfLhe3bt2PXrl2CnmtwcBDHjx/Hli1bMDU1xZG3sLAQU1NTAPzNI4BPjSWW6ijl05ro4SCRSFBfX4/6+nr89Kc/RWdnJ3bt2oWrr74aWVlZ2Lp1Ky6//PJ5IhFqXZWamoqKigpR6vB0zrvNZuNW8mDXz++rp7X6/v5+TnIaqVZPu9A2btwo2jAC6m4j5r6fwmazYdu2bbjzzjtxyy23iHIOq9WKa665Br/97W/nVQsitVgvZZyxROeDYRjU1NTg4Ycfxo9//GP09vZi165d2LZtG9RqNb785S9j69atkMlk+OCDD9DY2CiaIww/WojWsYVhGGRmZiIzMxNVVVWw2WyYmprC0aNHQ5am6JZg48aNoo0VWkiSOxwO3Hjjjbj55ptFI7nH48E111yDm266CVdffTUAoKCgABMTE1zoTg0rqHkEBd9YYini9J0DGycYhsHq1avx4IMP4sMPP8Rzzz0Hn8+HG264AVu2bMF7770HhmGiGjwRK1iWRXt7O6RSKdatWxfX6kCHN1ZWVmLLli1Ys2YNN+X1yJEjGB4exsDAAMbHx88YkjudTtx000249tpr8bWvfU2UcxBC8PWvfx3r1q3D9773Pe71K664An/5y18AzA552Lp1K/f6X//6VxBC8PHHHyMzM3PJhu3AaZ6MEwo6nQ4XX3wxfvzjH2NkZASvvPIK3G43Lr/8cmzdulWQHnmfz+fXcy8GnE4nurq6/DrKxDCNMJvN3EBFsQceuN1u3HLLLbjooovw7W9/W7TQ+f3338e5557r1+f/85//HFu2bMH111+P4eFhrFy5Ei+99BK3nbrnnnuwf/9+pKSk4M9//jM2bdoU7+mTWfeFACEE4+PjXOhFCMHU1BT+9re/4W9/+xvMZjMuu+wybN26Na42UaqNLywsFDW8GxgY4Ewp+C22TqeTK9vRMc3xYiFJ7vF4cPvtt+Pss8/G/ffff9ruj6NAkuhLAdPT03j11Vexe/du6HQ6XHLJJdi6dWtU4TdtThHTJJIOuHA6nUFtuLxeL/R6PaampmCz2aDRaFBQUBBzrX5mZgadnZ2iOejw4fV6sX37dtTX1+Ohhx46k0kOJIm+9GA0Grme+pGREVx88cW48sorg1o70cYRMfX31HmGtuRGIkSwvvpoavULSXKfz4e7774blZWVeOSRR850kgOnM9H379+P++67Dz6fD9u3b8cDDzwQ76GWLOjond27d6O3txdf+MIXcOWVV6KxsRG9vb3QarWoq6sTrXEk0bHI/Fq90WgMWavne8cvBMnvvfdeFBQU4LHHHlsOJAdOV6L7fD5UV1fjn//8J0pKStDc3IydO3eeDpY+ccNms2Hfvn1ce63L5cLPf/5zbN26VTR7qY6ODqhUKkGmvvJr9Xq9nkvmyeVy9Pb2iqqqo2BZFt/73veQlpaGX//616IJlZYgRCe6KJ/koUOHUFVVhYqKCigUCmzbtg179uwR41RLBqmpqbjuuutw7733Ij09HT/60Y/w5ptv4qyzzsL3vvc9/Otf/6Jm/QmDzj5PTU0VzHOd1upXr16NLVu2oLKyEgaDASdPnoRMJsP09DTcbrcAVx8cLMvigQcegEKhWG4kXxCIIpgJJg9MQPB/WmHlypXYt28fCgsLcccdd8DlcuHAgQN46aWXcP/99+Oss87ieurjqXHTMl1OTg7KyspEeAezpPd4PLBYLDjnnHPAsiy0Wi1OnjwpqAc+Bcuy+MlPfgK3242nn346SXIRsCyUcQuJQG83pVKJSy+9FJdeeik8Hg/effdd7Nq1Cz/84Q+xadMmbN26FRdccEFUOvSF8JADZptgenp6/PTx5eXlKC8v5/rq29vbE/LApyCE4NFHH4Ver8dzzz2XJLlIEOVTPd3kgQsFuVyOCy+8EE8//TROnjyJ22+/HW+//TbOPfdcbN++Ha+//jocDkfQv6VOrcXFxQtC8oaGhqAPHzq4sampiXsQ9PT04OOPP0Zvby/MZnPUqkJCCB5//HEMDw/j2WefFa1DcP/+/VizZg2qqqrwi1/8QpRzLHWIkozzer2orq7GgQMHsGLFCjQ3N2PHjh2ora0N+vuL3PS/6PD5fPj444+xa9cuHDhwAGvWrOF66lNTU2GxWNDZ2cnNihMLer0evb29cTXB0Fq9VquF1WqNWKsnhOC3v/0tTpw4gR07dogm1T1NEsOnZ9YdAPbt24fvfOc78Pl8+NrXvoZ///d/D/m7i9z0v6TAsiyOHj2Kl19+Gf/4xz9QXFyMnp4ePPPMM9i8ebNo502E5IFgWZYjfbBaPSEETz31FD744AO89NJLonXWAbPDFR955BG8+eabAIDHHnsMAPDggw+Kds44cPoSPREscNP/ksXw8DC+9KUvobm5Ga2trSgqKuJ66oV0bp2enkZ/fz8aGhoEJ11grX7Hjh1QKBQYHBzEq6++KlqPPMWuXbuwf/9+PPvsswCA//mf/8Enn3yCP/zhD6KeN0YsP+OJM7HpP16Mj4/jueeew1lnnQVCCDo6OrBr1y5ceeWV0Gg0XE99IpbQOp0OAwMDopAcmN9X//bbb+O1116DTCbDzTffjJ07dwo+ETaJ+VhSn/CZ2vQfLz7zmc9w/80wDGpra1FbW4uHH34YPT092LVrF66//nqkpKTgiiuuwBVXXIGCgoKoPydKcjHbWfnYsWMHPv74Y3zyySdQq9Xo7+8XneTJxPAslkwtI1zTP4DTuulfaDAMg+rqajz00EP46KOP8Oyzz8Lj8eCWW27BJZdcgieffBJjY2Nhs99arXZBSf7yyy/jhRdewJ49e5CSkgKGYVBZWSn6eZubm9HT04OBgQG43W68+OKLuOKKK0Q/71LDkiC60E3/Pp8PGzduxOWXXw5gtn1zy5YtqKqqwg033MApvFwuF2644QZUVVVhy5YtGBwcXKB3LBwYhkFFRQV+8IMf4P3338cLL7wAuVyO7du34+KLL8bvfvc7DA4O+pFeq9ViaGhowUj+6quv4tlnn8WePXtE85QLBZlMhj/84Q/44he/iHXr1uH6668PWf05k7EkknFCN/0/8cQTOHLkCNd0cv311+Pqq6/Gtm3bcNddd6G+vh533303nnrqKbS0tODpp5/Giy++iFdeeQX/93//txBvWXQQQjA5Ocn11FutVlx22WWQSqXweDz47ne/uyAkf+ONN/Cb3/wGb7zxhujz0E9jiL8nJYSE+99ph5GREfL5z3+eHDhwgFx22WWEZVmSk5NDPB4PIYSQDz/8kFx88cWEEEIuvvhi8uGHHxJCCPF4PCQnJ4ewLLto1y4mtFotueOOO0hhYSHZvHkz+fGPf0wOHz5MrFYrsdlsovzvlVdeIZs3bybT09OL/faXOiLxMOH/LalknBD4zne+g8cffxwWiwXAbH2Y7/1OM/SAf/ZeJpMhMzMTer1etMGGiwm5XI6JiQl0d3fD4/Hgtddew3/8x39gdHQUX/ziF3HllVdi/fr1gklQ33nnHTz66KN44403ROvFTyJ6LIk9ulDYu3cv8vPz0dTUtNiXsuSQlZWF1157Denp6dBoNLjtttvw+uuv4+DBg6irq8OvfvUrnHPOOfjxj3+Mo0ePgmXZuM/13nvv4eGHH8brr78uqpIviehxRq3oH3zwAV577TXs27cPTqcTZrMZ9913H0wmE7xeL2QymV+GnmbvS0pK4PV6MTMzs+xWn8zMTNx000246aabYLVasW/fPvzhD39AR0cHLrjgAlx55ZVobm6OWof+0Ucf4YEHHsDevXtFm+ueRByIENuftnjnnXfIZZddRggh5NprryU7d+4khBBy5513kieffJIQQsgf/vAHcueddxJCCNm5cye57rrrFudilyDsdjt55ZVXyM0330xqa2vJXXfdRfbv309mZmZC7skPHjxI6uvrydDQ0GJf/ukG0ffoy4LofX19pLm5mVRWVpJrr72WOJ1OQgghDoeDXHvttaSyspI0NzeTvr4+v2MYjUZyzTXXkDVr1pC1a9eSDz/8kOj1enLhhReSqqoqcuGFFxKDwUAIIYRlWfLtb3+bVFZWkrq6OnL06NGFfcMiwul0kr1795LbbruN1NTUkO3bt5O9e/cSk8nEkfz9998nGzZsIP39/Yt9uacjkkRfTNx6663kmWeeIYQQ4nK5iNFoJD/4wQ/IY489Rggh5LHHHiP/9m//Rggh5I033iBf+tKXCMuy5KOPPiKbN29etOsWE263m/zjH/8gd9xxB6mpqSFf/epXyRNPPEHWr19Purq6FvvyTlckib5YMJlMpLy8fF65rbq6moyPjxNCCBkfHyfV1dWEEELuuOMOsmPHjqC/d6bC4/GQd955h3z2s58lBw8eXOzLOZ0hOtHPqKy7kBgYGEBeXh5uv/12bNy4Edu3b+dmnsXSaHMmQyaT4fzzz8d7772Hz33uc4t9OUmEQZLoIeD1enHs2DHcfffdOH78OFJTU+e5kyzHRpuFwA9+8AOsXbsWGzZswFVXXQWTycT97LHHHkNVVRXWrFnD9ZgDSReZSEgSPQRKSkpQUlKCLVu2AACuvfZaHDt2LNloswC46KKL0NbWhpaWFlRXV3NmER0dHXjxxRfR3t6O/fv345vf/CZ8Ph98Ph++9a1v4e9//zs6Ojqwc+dOdHR0LPK7WFpIEj0ECgsLUVpaiq6uLgDAgQMHUFNTE3ejzW9+8xvU1tZi/fr1uPHGG+F0Os/oZptEcPHFF3NKxs985jMYHR0FAOzZswfbtm2DUqnEqlWrUFVVhUOHDi1Le/FYkSR6GPzXf/0XbrrpJmzYsAEnTpzAQw89hAceeAD//Oc/sXr1arz11lvcBJpLL70UFRUVqKqqwje+8Q089dRT3HHGxsbw+9//HkeOHEFbWxt8Ph9efPFF/PCHP8R3v/td9Pb2Ijs7G8899xwA4LnnnkN2djZ6e3vx3e9+Fz/84Q8X5f0vBfzpT3/CJZdcAiB0HmQ55kdixRmljBMaDQ0NOHLkyLzXDxw4MO81hmHw5JNPhjyW1+uFw+GAXC6H3W5HUVER3n77bezYsQMA8NWvfhWPPPII7r77buzZswePPPIIgNktwz333ANCyBmVD7jwwgsxOTk57/VHH32Ui5IeffRRyGQy3HTTTQt9eWcckkRfAKxYsQLf//73UVZWBrVajYsvvhhNTU3LutnmrbfeCvvz559/Hnv37sWBAwe4B1y4PEgyPxIeydB9AWA0GrFnzx4MDAxgfHwcNpsN+/fvX+zLWrLYv38/Hn/8cbz22mt+M9ivuOIKvPjii3C5XBgYGEBPTw82b96cdJGJAskVfQHw1ltvYdWqVcjLywMAXH311fjggw+SzTYhcM8998DlcuGiiy4CMJuQe/rpp1FbW4vrr78eNTU1kMlkePLJJ7lmG+oiQ+3Fl6OLTFhEUNQkIQA+/vhjUlNTQ2w2G2FZltx6663k97//fUzNNrfffjvJy8sjtbW13HHj0d0///zzpKqqilRVVZHnn39+oT6CJMIjKYE9U/Dwww+TNWvWkNraWnLzzTcTp9MZU7PNu+++S44ePepH9Fh193q9nqxatYro9XpiMBjIqlWruIdDEouKJNGT+BQDAwN+RI9Vd79jxw5yxx13cK8H/l4Si4ak1j2J0IhVd5+sNy9fRHKBTWIJgWGYcgB7CSHr5/5tIoRk8X5uJIRkMwyzF8AvCCHvz71+AMAPAZwPQEUI+f/mXv8xAAch5NcL+kaSWHAkV/TTG1MMwxQBwNz/a+deHwNQyvu9krnXQr2exBmOJNFPb7wG4Ktz//1VAHt4r9/KzOIzAGYIIRMA3gRwMcMw2QzDZAO4eO61JM5wJOvopwkYhtmJ2dA7l2GYUQA/AfALAC8xDPN1AEMArp/79X0ALgXQC8AO4HYAIIQYGIb5GYDDc7/3U0KIYcHeRBKLhuQePYkklgGSoXsSSSwDJImeRBLLAEmiJ5HEMkCS6EkksQyQJHoSSSwDJImeRBLLAEmiJ5HEMkCS6EkksQzw/wMimwpjoGrWGQAAAABJRU5ErkJggg=="
class="
jp-needs-light-background
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 id="Advanced-Usage-and-Concepts">Advanced Usage and Concepts<a class="anchor-link" href="#Advanced-Usage-and-Concepts">&#182;</a></h1>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Calculating-AEP">Calculating AEP<a class="anchor-link" href="#Calculating-AEP">&#182;</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Calculating AEP in FLORIS V3 takes advantage of the new vectorized framework to substantially reduce the computation time with respect to V2.4.</p>
<p>In these examples we demonstrate a simplied AEP calculation for a 25-turbine farm using several different modeling options.</p>
<p>We will make a simplifying assumption that every wind speed and direction is equally likely.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">wind_directions</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">360.</span><span class="p">,</span> <span class="mf">5.</span><span class="p">)</span>
<span class="n">wind_speeds</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mf">5.</span><span class="p">,</span> <span class="mf">25.</span><span class="p">,</span> <span class="mf">1.</span><span class="p">)</span>

<span class="n">num_wind_directions</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">wind_directions</span><span class="p">)</span>
<span class="n">num_wind_speeds</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">wind_speeds</span><span class="p">)</span>
<span class="n">num_bins</span> <span class="o">=</span> <span class="n">num_wind_directions</span> <span class="o">*</span> <span class="n">num_wind_speeds</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Calculating AEP for </span><span class="si">%d</span><span class="s1"> wind direction and speed combinations...&#39;</span> <span class="o">%</span> <span class="n">num_bins</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Calculating AEP for 1440 wind direction and speed combinations...
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Set up a square 25 turbine layout</span>
<span class="n">N</span> <span class="o">=</span> <span class="mi">5</span>  <span class="c1"># Number of turbines per row and per column</span>
<span class="n">D</span> <span class="o">=</span> <span class="mf">126.</span> 

<span class="n">X</span><span class="p">,</span> <span class="n">Y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">meshgrid</span><span class="p">(</span>
    <span class="mf">7.0</span> <span class="o">*</span> <span class="n">D</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span>
    <span class="mf">7.0</span> <span class="o">*</span> <span class="n">D</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span>
<span class="p">)</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">flatten</span><span class="p">()</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">Y</span><span class="o">.</span><span class="n">flatten</span><span class="p">()</span>
<span class="n">num_turbine</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of turbines = </span><span class="si">%d</span><span class="s2">&quot;</span> <span class="o">%</span> <span class="n">num_turbine</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Number of turbines = 25
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Define several models</span>
<span class="n">fi_jensen</span> <span class="o">=</span> <span class="n">FlorisInterface</span><span class="p">(</span><span class="s2">&quot;inputs/jensen.yaml&quot;</span><span class="p">)</span>
<span class="n">fi_gch</span> <span class="o">=</span> <span class="n">FlorisInterface</span><span class="p">(</span><span class="s2">&quot;inputs/gch.yaml&quot;</span><span class="p">)</span>
<span class="n">fi_cc</span> <span class="o">=</span> <span class="n">FlorisInterface</span><span class="p">(</span><span class="s2">&quot;inputs/cc.yaml&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Assign the layouts, wind speeds and directions</span>
<span class="n">fi_jensen</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span><span class="n">layout</span><span class="o">=</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">),</span> <span class="n">wind_directions</span><span class="o">=</span><span class="n">wind_directions</span><span class="p">,</span> <span class="n">wind_speeds</span><span class="o">=</span><span class="n">wind_speeds</span><span class="p">)</span>
<span class="n">fi_gch</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span><span class="n">layout</span><span class="o">=</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">),</span> <span class="n">wind_directions</span><span class="o">=</span><span class="n">wind_directions</span><span class="p">,</span> <span class="n">wind_speeds</span><span class="o">=</span><span class="n">wind_speeds</span><span class="p">)</span>
<span class="n">fi_cc</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span><span class="n">layout</span><span class="o">=</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">),</span> <span class="n">wind_directions</span><span class="o">=</span><span class="n">wind_directions</span><span class="p">,</span> <span class="n">wind_speeds</span><span class="o">=</span><span class="n">wind_speeds</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Calculate the AEP and use the jupyter time command to show computation time:</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%%time</span>
<span class="n">fi_jensen</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">()</span>
<span class="n">jensen_aep</span> <span class="o">=</span> <span class="n">fi_jensen</span><span class="o">.</span><span class="n">get_farm_power</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">/</span> <span class="n">num_bins</span>  <span class="o">/</span> <span class="mf">1E9</span> <span class="o">*</span> <span class="mi">365</span> <span class="o">*</span> <span class="mi">24</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>CPU times: user 3.11 s, sys: 156 ms, total: 3.27 s
Wall time: 2.5 s
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%%time</span>
<span class="n">fi_gch</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">()</span>
<span class="n">gch_aep</span> <span class="o">=</span> <span class="n">fi_gch</span><span class="o">.</span><span class="n">get_farm_power</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">/</span> <span class="n">num_bins</span>  <span class="o">/</span> <span class="mf">1E9</span> <span class="o">*</span> <span class="mi">365</span> <span class="o">*</span> <span class="mi">24</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>CPU times: user 4.81 s, sys: 78.1 ms, total: 4.89 s
Wall time: 4.04 s
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># %%time</span>
<span class="c1"># fi_cc.calculate_wake()</span>
<span class="c1"># cc_aep = fi_cc.get_farm_power().sum() / num_bins  / 1E9 * 365 * 24</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Show the results</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Jensen </span><span class="si">%.1f</span><span class="s1"> GWh&#39;</span> <span class="o">%</span> <span class="n">jensen_aep</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;GCH </span><span class="si">%.1f</span><span class="s1"> GWh&#39;</span> <span class="o">%</span> <span class="n">gch_aep</span><span class="p">)</span>
<span class="c1"># print(&#39;CC %.1f GWh&#39; % cc_aep)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Jensen 843.2 GWh
GCH 843.9 GWh
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Wake-Steering-Design">Wake Steering Design<a class="anchor-link" href="#Wake-Steering-Design">&#182;</a></h2>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>FLORIS V3 further includes new optimization routines for the design of wake steering controllers.  The SerialRefine is a new method for quickly identifying optimum yaw angles.</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Demonstrate on 7-turbine single row farm</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">6</span><span class="o">*</span><span class="mi">7</span><span class="o">*</span><span class="n">D</span><span class="p">,</span> <span class="mi">7</span><span class="p">)</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros_like</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
<span class="n">wind_speeds</span> <span class="o">=</span> <span class="p">[</span><span class="mf">8.</span><span class="p">]</span>
<span class="n">wind_directions</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">360.</span><span class="p">,</span> <span class="mf">2.</span><span class="p">)</span>
<span class="n">fi_gch</span><span class="o">.</span><span class="n">reinitialize</span><span class="p">(</span><span class="n">layout</span><span class="o">=</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">),</span> <span class="n">wind_directions</span><span class="o">=</span><span class="n">wind_directions</span><span class="p">,</span> <span class="n">wind_speeds</span><span class="o">=</span><span class="n">wind_speeds</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">floris.tools.optimization.yaw_optimization.yaw_optimizer_sr</span> <span class="kn">import</span> <span class="n">YawOptimizationSR</span>

<span class="c1"># Define the SerialRefine optimization</span>
<span class="n">yaw_opt</span> <span class="o">=</span> <span class="n">YawOptimizationSR</span><span class="p">(</span>
    <span class="n">fi</span><span class="o">=</span><span class="n">fi_gch</span><span class="p">,</span>
    <span class="n">minimum_yaw_angle</span><span class="o">=</span><span class="mf">0.0</span><span class="p">,</span>  <span class="c1"># Allowable yaw angles lower bound</span>
    <span class="n">maximum_yaw_angle</span><span class="o">=</span><span class="mf">25.0</span><span class="p">,</span>  <span class="c1"># Allowable yaw angles upper bound</span>
    <span class="n">Ny_passes</span><span class="o">=</span><span class="p">[</span><span class="mi">5</span><span class="p">,</span> <span class="mi">4</span><span class="p">],</span>
    <span class="n">exclude_downstream_turbines</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">exploit_layout_symmetry</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
<span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%%time</span>
<span class="c1">## Calculate the optimum yaw angles for 25 turbines and 72 wind directions</span>
<span class="n">df_opt</span> <span class="o">=</span> <span class="n">yaw_opt</span><span class="o">.</span><span class="n">optimize</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>[Serial Refine] Processing pass=0, turbine_depth=0 (0.0 %)
[Serial Refine] Processing pass=0, turbine_depth=1 (7.1 %)
[Serial Refine] Processing pass=0, turbine_depth=2 (14.3 %)
[Serial Refine] Processing pass=0, turbine_depth=3 (21.4 %)
[Serial Refine] Processing pass=0, turbine_depth=4 (28.6 %)
[Serial Refine] Processing pass=0, turbine_depth=5 (35.7 %)
[Serial Refine] Processing pass=0, turbine_depth=6 (42.9 %)
[Serial Refine] Processing pass=1, turbine_depth=0 (50.0 %)
[Serial Refine] Processing pass=1, turbine_depth=1 (57.1 %)
[Serial Refine] Processing pass=1, turbine_depth=2 (64.3 %)
[Serial Refine] Processing pass=1, turbine_depth=3 (71.4 %)
[Serial Refine] Processing pass=1, turbine_depth=4 (78.6 %)
[Serial Refine] Processing pass=1, turbine_depth=5 (85.7 %)
[Serial Refine] Processing pass=1, turbine_depth=6 (92.9 %)
CPU times: user 2.28 s, sys: 266 ms, total: 2.55 s
Wall time: 2.05 s
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>In the results, T0 is the upstream turbine when wind direction is 270, while T6 is upstream at 90 deg</p>

</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Show the results</span>
<span class="n">yaw_angles_opt</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">vstack</span><span class="p">(</span><span class="n">df_opt</span><span class="p">[</span><span class="s2">&quot;yaw_angles_opt&quot;</span><span class="p">])</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">axarr</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">),</span> <span class="mi">1</span><span class="p">,</span> <span class="n">sharex</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">10</span><span class="p">))</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">)):</span>
    <span class="n">axarr</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">wind_directions</span><span class="p">,</span> <span class="n">yaw_angles_opt</span><span class="p">[:,</span> <span class="n">i</span><span class="p">],</span> <span class="s1">&#39;k-&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;T</span><span class="si">%d</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="n">i</span><span class="p">)</span>
    <span class="n">axarr</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Yaw (Deg)&#39;</span><span class="p">)</span>
    <span class="n">axarr</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
    <span class="n">axarr</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">grid</span><span class="p">(</span><span class="kc">True</span><span class="p">)</span>
<span class="n">axarr</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Wind Direction (Deg)&#39;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">



<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>Text(0.5, 0, &#39;Wind Direction (Deg)&#39;)</pre>
</div>

</div>

<div class="jp-OutputArea-child">



<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAl4AAAJNCAYAAADgY3uzAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAACE0ElEQVR4nOzde3yU9Zn//9dFCAmQgBgwclATlaPahoPg2aDiqdZzPSuxbul2e5D91rb2a7fSdrt1d7v9ud3WftfdCnhk1drq2lprlahURUFQOYhQBI2CYhBzgEAI1++PmWDEuSeTydz3ZDLv5+Mxj8zc99z3fc3FPeHK5/O5P7e5OyIiIiISvj7ZDkBEREQkX6jwEhEREYmICi8RERGRiKjwEhEREYmICi8RERGRiKjwEhEREYlI32wHkIqhQ4d6RUVFqMdobm5m4MCBoR4jVyk3wZSbYMpNMOUmMeUlmHITrCfmZunSpR+4+7BE63Ki8KqoqGDJkiWhHqO2tpbq6upQj5GrlJtgyk0w5SaYcpOY8hJMuQnWE3NjZhuD1qmrUURE8tJDDz0U+h/1IvtS4SUiInlnz549XHLJJUybNo1vf/vbtLS0ZDskyRMqvEREJO/s2LGDtrY2Kisr+dd//Veqqqp45513sh2W5IGcGOMlIiKSSU1NTQB885vf5NBDD+XMM8/k3nvv5Vvf+laWI8s/ra2t1NXVpd3qOHjwYFavXp3hqFJTXFzMqFGjKCwsTHkbFV4iIpJ32guvkpISzjjjDCorK1m8eHGWo8pPdXV1lJaWUlFRgZl1efvGxkZKS0tDiCw5d6e+vp66ujoqKytT3k5djSIikncaGxuBWOEFMG3aNBVeWdLS0kJZWVlaRVc2mRllZWVdbqlT4SUiInmnY4sXxAqvuro63n333WyGlbdyrehql07cKRVeZjbEzI4ws0PNTMWaiIjktESFF6BWrzxTX19PVVUVVVVVHHjggYwcOXLv60ceeYSxY8dy+OGHc8stt2TsmIFjvMxsMPBV4HKgH7AFKAbKzewF4DZ3X5ixSERERCKyb+E1ceJECgsLWbx4MRdccEE2Q5MIlZWVsXz5cgDmzJlDSUkJN9xwA21tbYwZM4YnnniCUaNGcfTRR3PuuecyYcKEbh8zWevVg8DbwInuPtbdT3D3Ke5+EHALcJ6ZXRe0sZkdZGYLzWyVma00s+vjy/c3syfMbG3855BufwoREZEu2LfwKi4upqqqSi1eAsCLL77I4YcfzqGHHkq/fv247LLLePjhhzOy78DCy91nuPtd7r4twbql7j7b3X+dZN+7gW+6+wTgGOCrZjYBuBF40t1HA0/GX4uIiERm38ILYt2NS5Ysoa2tLVthSQ/xzjvvcNBBB+19PWrUqIzN89bpdBJmNinB4o+Aje6+O2g7d98EbIo/bzSz1cBI4DygOv62+UAt8J0uRS0iItINQYXXL37xC1atWsVRRx2VrdDy2uzZs/d2/aWqra2NgoKCwPVVVVXceuut3Qssg1IZKH8b8AJwO/BfwPPAA8AaMzs9lYOYWQUwEVgMlMeLMoDNQHkXYxYREemWpqYm+vTpQ3Fx8d5lGmAv7UaOHMnbb7+993VdXR0jR47MyL5TmUD1XeA6d18JEO8u/CHwbeAh4E/JNjazEuA3wGx3b+h46aW7u5l5wHazgFkA5eXl1NbWphBq+pqamkI/Rq5SboIpN8GUm2DKTWJR5uX111+nf//+PP3003uXuTuDBg3id7/7HYcffngkcaSqN58zgwcP3juv2o9+9KMub99Zixd8PG9bMjt37qSwsJDGxkbGjRvHG2+8wWuvvcaIESO49957+fWvf51wPy0tLV36t0ml8BrTXnQBuPsqMxvn7us7m7/CzAqJFV33uPtD8cXvmdlwd99kZsOB9xNt6+63E2tlY8qUKV5dXZ1CqOmrra0l7GPkKuUmmHITTLkJptwkFmVe7r77bgYPHvyp4x133HG89dZbPe7fpzefM6tXr+7WzPOZmrm+qKiIoqKivfv65S9/yUUXXURbWxtf/OIXmTp1asLtiouLmThxYsrHSaXwWmlmvwIWxF9fCqwysyKgNWgji1VlvwZWu/vPOqx6BJhJ7MrImUBmLhMQERFJUVNT0yfGd7WbNm0aP/rRjwLXS+81Z86cT7w+++yzOfvsszN+nFTGeNUA64DZ8cf6+LJWYHqS7Y4HrgZOMbPl8cfZxAquGWa2Fjgt/lpERCQyQYXVsccey549e3jhhReyEJXkg05bvNx9h5ndBjzq7mv2Wd2UZLtFQFBf5KmphygiIpJZQYXXcccdR0FBAbW1tZx22mlZiEx6u05bvMzsXGA58Mf46yozeyTkuEREREITVHiVlpYyZcqUXjuQXbIvla7Gm4GpwDYAd18OVIYXkoiISLiSjeGqrq7mxRdfpLm5OeKo8pd7wgkOerx04k6l8Gp194/2PVaXjyQiItJDdFZ4tba28vzzz0ccVX4qLi6mvr4+54ovd6e+vv4Tc8GlItWrGq8ACsxsNPAN4Lk0YhQREekRkhVeJ5xwAgUFBSxcuFDjvCIwatQo6urq2LJlS1rbt7S0dLn4yZTi4mJGjRrVpW1SKby+DtwE7ATuAx4Huj7DmYiISA/g7jQ1NQXO/VRSUsLRRx+tcV4RKSwspLIy/RFMtbW1XZpHK9s67Wp09+3ufpO7H+3uU+LPW6IITkREJNN27txJW1tb0nm6NM5LwpK08DKzmWb2spk1xx9LzOyaqIITERHJtEQ3yN5XdXU1u3fv5rnnNLJGMiuw8DKzmcQmTP0mMAIYSez+jNeb2dWRRCciIpJhqRRexx9//N75vEQyKVmL11eAC9x9obt/5O7b3P0p4CLgq9GEJyIiklmpFF7t47wWLlwYVViSJ5IVXoPcfcO+C+PLBoUVkIiISJhSKbwApk+fzksvvUR9fX0UYUmeSFZ47UhznYiISI+VauF16aWXsnv3bhYsWBBFWJInkhVe483s1QSP14BxUQUoIiKSSY2NjUDnhddnP/tZqqqqmDdvXgRRSb5INo/X+MiiEBERiUiqLV4ANTU1zJ49mxUrVnDkkUeGHZrkgWQtXm+5+8agB4CZWURxioiIZERXCq8rrriCvn37Mn/+/LDDkjyRrPBaaGZfN7ODOy40s35mdoqZzQdmhhueiIhIZnWl8Bo2bBjnnHMOd911F62trWGHJnkgWeF1JtAG3Gdm75rZKjNbD6wFLgdudfd5QRub2R1m9r6ZreiwbH8ze8LM1sZ/DsnQ5xAREUlJe+E1YMCAlN5/7bXX8t577/H444+HGZbkicDCy91b3P02dz8eOAQ4FZjk7oe4+5fcfVkn+55HrHjr6EbgSXcfDTwZfy0iIhKZpqYmBg4cSJ8+nd41D4CzzjqLYcOGMXfu3JAjk3yQ0lnn7q3uvsndt6W6Y3d/Bti6z+LzgPaO8vnA+anuT0REJBOamppS6mZsV1hYyLnnnsvTTz8dYlSSL1Ir9zOn3N03xZ9vBsojPr6IiOS5rhZeAOPHj6e+vl6TqUq3JZtOIlTu7mbmQevNbBYwC6C8vDz0+2U1NTXpnlwBlJtgyk0w5SaYcpNYVHnZsGEDQJeOtWvXLgAWLFjAEUccEUJUyemcCZZruem08DKz64Bn3H1tBo73npkNd/dNZjYceD/oje5+O3A7wJQpU7y6ujoDhw9WW1tL2MfIVcpNMOUmmHITTLlJLKq8FBcXc+CBB3bpWCNGjOD//t//S0lJSVb+7XTOBMu13KTS1Xgw8J9mtt7MHohPMVGV5vEe4eMpKGYCD6e5HxERkbSk09VYWVlJ3759WbNmTUhRSb7otPBy95vd/RTgCOBZ4FvA0s62M7P7gOeBsWZWF285uwWYYWZrgdPir0VERCKTTuFVWFjIoYceyhtvvBFSVJIvUulq/B5wPFACLANuIFaAJeXulwesOrUrAYqIiGRSOoUXwNixY9XiJd2WSlfjhUAZ8GfgIeDhDlcmioiI5JR0C68xY8awdu1a9uzZE0JUki9S6WqcRKxb8EVgBvCamS0KOzAREZEwdKfFa+fOnbz11lshRCX5IpWuxiOBE4GTgSnA26TQ1SgiItLT7Nq1i127dlFaWtrlbceMGQPAG2+8QUVFRYYjk3yRSlfjLcAg4OfAeHef7u7fDzcsERGRzGtubgZSu0H2vsaOHQugcV7SLZ22eLn7OVEEIiIiErb2G2SnU3iVl5dTWlqqKxulW1LpahwN/ASYABS3L3f3Q0OMS0REJOO6U3iZma5slG5LpatxLvArYDcwHbgTuDvMoERERMLQncILYuO81OIl3ZFK4dXf3Z8EzN03uvsc4HPhhiUiIpJ53S28xo4dy1tvvcWOHTsyGZbkkcDCy8z+YGaVwE4z6wOsNbOvmdkFxCZTFRERySmZaPFyd9atW5fJsCSPJGvxmgs8DjwGDAS+AUwGrgKuCT80ERGRzMpEixfoykZJX2Dh5e4PAJOItW4tAi4DVgDPEbuFkIiISE7pbuE1evRoAI3zkrR1dlXjLqAZKAJKAd0nQUREclZ3C6+SkhJGjhzJ6tWrMxmW5JHAwsvMzgR+BjwCTHL37ZFFJSIiEoLGxkYABg4cmPY+TjjhBP7whz+wc+dOioqKMhWa5IlkY7xuAr7g7jeq6BIRkd6gqamJ4uJi+vbtdBrLQDU1NWzdupVHH300g5FJvkg2xutEd18ZZTAiIiJhSvcG2R3NmDGDESNGMG/evMwEJXkllXm8Ms7MzjSzNWa2zsxuzEYMIiKSfzJReBUUFHD11Vfz2GOPsXnz5gxFJvki8sLLzAqAXwJnEbsN0eVmNiHqOEREJP9kovCCWHdjW1sbd9+tG7lI12SjxWsqsM7d17v7LmABcF4W4hARkTyTqcJr3LhxHHPMMcybNw93z0Bkki/SH12YvpHA2x1e1wHTshDHXmvWrGHJkiW0trZmM4we65VXXlFuAig3wZSbYMpNYlHk5Z133mHEiBEZ2de1117Ll7/8ZebOnctBBx2UkX0G0TkTrKu5mTp1KoMHDw4xouSyUXilxMxmAbMAysvLqa2tDe1Yv/jFL/jNb34T2v5FRKTnGDlyZEb+TxkxYgTFxcVcd9113Q9KInPbbbcxfvz4rB0/G4XXO0DHPw1GxZd9grvfDtwOMGXKFK+urg4toIMPPpiTTz6ZSZMmhXaMXPbyyy8rNwGUm2DKTTDlJrGo8nLUUUcxaNCgjOxr1apVvPvuuxnZVzI6Z4J1NTef+cxnKC0tDTGi5LJReL0EjI7fgPsdYrciuiILcex16KGHctRRR3H88boTUiKtra3KTQDlJphyE0y5SSwX81JZWUllZWXox8nF3EQl13Jj2RgUaGZnA7cCBcAd7v7jTt6/BdgYclhDgQ9CPkauUm6CKTfBlJtgyk1iyksw5SZYT8zNIe4+LNGKrBRePZGZLXH3KdmOoydSboIpN8GUm2DKTWLKSzDlJliu5SYrE6iKiIiI5CMVXiIiIiIRUeH1sduzHUAPptwEU26CKTfBlJvElJdgyk2wnMqNxniJiIiIREQtXiIiIiIRUeElIiIiEhEVXiIiIiIRUeElIiIiEhEVXiIiIiIRUeElIiIiEhEVXiIiIiIRUeElIiIiEpG+2Q4gFUOHDvWKiopQj9Hc3MzAgQNDPUauUm6CKTfBlJtgyk1iyksw5SZYT8zN0qVLP3D3YYnW5UThVVFRwZIlS0I9Rm1tLdXV1aEeI1cpN8GUm2DKTTDlJjHlJZhyE6wn5sbMNgatU1ejiIjkpWnTpvGrX/0q22FInlHhJSIieWfXrl28+OKLfO9736OxsTHb4UgeUeElIiJ5p7m5GYCtW7fyH//xH1mORvJJTozxEhERyaSmpiYAioqK+Ld/+ze+9rWvMWjQoCxHlX9aW1upq6ujpaUl7X0MHjyY1atXZzCq1BUXFzNq1CgKCwtT3kaFl4iI5J32wuvrX/86P/3pT/mP//gPbrrppixHlX/q6uooLS2loqICM0trH42NjZSWlmY4ss65O/X19dTV1VFZWZnydupqFBGRvNNeeFVXV3POOefwb//2bzQ0NGQ5qvzT0tJCWVlZ2kVXNpkZZWVlXW6tU+ElIiJ5p73wKikp4R/+4R/48MMPefDBB7McVX7KxaKrXTqxp9TVaGZDgBHADmCDu+/p8pFERER6iI6F16RJkxgyZAgvvPACX/ziF7McmUSpvr6eU089FYDNmzdTUFDAsGGxeU8nTZrEo48+ygEHHMCKFSsydszAwsvMBgNfBS4H+gFbgGKg3MxeAG5z94UZi0RERCQiHQsvM2Pq1KksXrw4y1FJ1MrKyli+fDkAc+bMoaSkhBtuuAGAZ555hq997Wtcc801GT1msq7GB4G3gRPdfay7n+DuU9z9IOAW4Dwzuy5oYzM7yMwWmtkqM1tpZtfHl+9vZk+Y2dr4zyEZ/UQiIiKd6Fh4QWwy1RUrVuxdLnLSSSex//77Z3y/gS1e7j4jybqlwNJO9r0b+Ka7v2xmpcBSM3sCqAGedPdbzOxG4EbgO12OXEREJE2JCq89e/awdOlSTj755GyGlrdmz569t/WpK9ra2igoKEi4rqqqiltvvbV7gWVYp4PrzWxSgsdhZpZ0fJi7b3L3l+PPG4HVwEjgPGB+/G3zgfO79QlERES6qL3war+58tSpUwHU3SihS2Vw/W3AJOBVwIAjgZXAYDP7irv/qbMdmFkFMBFYDJS7+6b4qs1AeRpxi4iIpK2pqYni4mL69o39Nzh06FAOO+wwFV5ZlG7LVLbm8UpXKoXXu8B17r4SwMwmAD8Evg08BCQtvMysBPgNMNvdGzpeeunubmYesN0sYBZAeXk5tbW1KYSavqamptCPkauUm2DKTTDlJphyk1iUeVmzZg1FRUWfOF5FRQXPPPNMj/y36a3nzODBg7t9r8y2traM3G9z586dFBYWfmJfTU1N7NmzJ+n+W1pauvRvk0rhNaa96AJw91VmNs7d13c2f4WZFRIruu5x94fii98zs+HuvsnMhgPvJ9rW3W8HbgeYMmWKV1dXpxBq+mprawn7GLlKuQmm3ARTboIpN4lFmZe5c+cyZMiQTxzv1Vdf5cknn2T06NGMHDkykjhS1VvPmdWrV3e7tSpTLV5FRUUUFRXt3dfll19ObW0tH3zwAePHj+cHP/gB11336WsKi4uLmThxYsrHSaXwWmlmvwIWxF9fCqwysyKgNWgji1VlvwZWu/vPOqx6BJhJ7MrImcDDKUcrIiKSAU1NTXsH1rebNm0aEBvndeGFF2YjLMmiOXPmfOL1fffdF8pxUpm5vgZYB8yOP9bHl7UC05NsdzxwNXCKmS2PP84mVnDNMLO1wGnx1yIiIpFJVHhVVVXRr18/jfOSUHXa4uXuO8zsNuBRd1+zz+rACU/cfRGxwfiJnJp6iCIiIpmVqPAqKiqiqqpKhZeEKpXpJM4FlgN/jL+uMrNHQo5LREQkNIkKL4h1Ny5ZsoS2trYsRCX5IJWuxpuBqcA2AHdfDlSGF5KIiEi4GhsbExZexxxzDM3NzSxd2tkc4ZIp7gknN8gJ6cSeSuHV6u4f7XusLh9JRESkhwhq8Tr77LMpKirirrvuykJU+ae4uJj6+vqcLL7cnfr6eoqLi7u0XapXNV4BFJjZaOAbwHNpxCgiItIjBBVe++23HxdccAH33nsvP/3pTykqKspCdPlj1KhR1NXVsWXLlrT30dLS0uXiJ1OKi4sZNWpUl7ZJpfD6OnATsBO4D3gc+FGXoxMREekB2tra2LFjR8LCC6CmpoYFCxbwv//7v1x88cURR5dfCgsLqazs3uil2traLs2jlW2ddjW6+3Z3v8ndj3b3KfHnLVEEJyIikmnNzc0AgYXXaaedxsiRI5k3b16EUUm+SFp4mdlMM3vZzJrjjyVmdk1UwYmIiGRa+w2ygwqvgoICrrnmGv74xz+yadOmhO8RSVdg4WVmM4lNmPpNYAQwktj9Ga83s6sjiU5ERCTDOiu8INbd2NbWxt133x1VWJInko3x+gpwgbtv6LDsKTO7iNjtg3TJh/QqO3bs4Lvf/S4NDQ0AlJWV8U//9E8UFhZmOTIRyaRUCq8xY8Zw3HHHMW/ePG644QY6uzexSKqSFV6D9im6AHD3DWY2KLyQRLLjscce49///d8ZPnw4AJs2beKYY47hoosuynJkIpJJqRReEGv1mjVrFkuWLOHoo4+OIjTJA8nGeO1Ic51ITqqtrWXAgAFs2LCBt99+mxEjRmhwrUgvlGrhdckll9C/f3/9HpCMSlZ4jTezVxM8XgPGRRWgSFRqa2s5/vjj6devHwUFBVx99dU89thjbN68OduhiUgGpVp4DR48mAsvvJB7772XlhZdzC+ZkbTwAj6f4HEOMCH80ESi88EHH/Daa69RXV29d1n74Np77rkne4GJSMa1F16lpaWdvrempoZt27bxyCO6RbFkRrLC6y133xj0ADCNNpRe4umnnwZg+vTpe5eNGzeOY445hrlz5+bk7SxEJLFUW7wg9jvhoIMOUnejZEyywmuhmX3dzA7uuNDM+pnZKWY2H5gZbngi0Wgf3zVlypRPLK+pqWHlypW6Ya5IL9KVwqt9Tq/HH3+cd955J+zQJA8kK7zOBNqA+8zsXTNbZWbrgbXA5cCt7j4vaGMzu8PM3jezFR2W7W9mT5jZ2vjPIRn6HCLdUltbywknnPCpqSMuvfRSiouL9deuSC/S1NREYWEh/fr1S+n9NTU17NmzR3N6SUYEFl7u3uLut7n78cAhwKnAJHc/xN2/5O7LOtn3PGLFW0c3Ak+6+2jgyfhrkazasmULK1as+MT4rnbtN8y97777aGtriz44Ecm4oBtkBzn88MM5/vjjue+++0KMSvJFp/dqBHD3Vnff5O7bUt2xuz8DbN1n8XnA/Pjz+cD5qe5PJCzPPPMMQMLCC+Dzn/88W7duZfny5dEFJSKh6WrhBXDCCSewatUqdu/eHVJUki9SKrwyqNzd2298tRkoj/j4Ip9SW1vLwIEDPzW+q93JJ5+8930ikvvSKbzGjh1La2srGzZsCCcoyRvJZq4Plbu7mQVeKmZms4BZAOXl5aH/p9fU1KT/WAP09tz8/ve/Z8KECfzlL38JfM9BBx3Eb37zGyZPnvyJ5b09N92h3ARTbhKLKi9vvfUW7t6lY7UPyH/wwQc55phjQoos+fF1ziSWc7lx96QP4DpgdGfvC9i2AljR4fUaYHj8+XBgTSr7mTx5sodt4cKFoR8jV/Xm3DQ0NDjgP/rRj5K+b9asWT5o0CBvbW39xPLenJvuUm6CKTeJRZWXE0880adPn96lbbZs2eKA/+xnPwspquR0zgTribkBlnhATZNKV+PBwH+a2XozeyA+xURVmnXeI3w8BcVM4OE09yOSEe3dBmPGjEn6vurqahoaGjTOS6QXSKersaysjCFDhvDGG2+EFJXki04LL3e/2d1PAY4AngW+BXQ6qZGZ3Qc8D4w1szozuw64BZhhZmuB0+KvRbKmvfCqqKhI+r72gfc51ZwtIgmlU3iZGWPHjmXNmjUhRSX5otPCy8y+Z2aPAX8CDgduAEZ1tp27X+7uw9290N1Hufuv3b3e3U9199Hufpq773vVo0ikUi28hg8fztixY1V4ifQC6RReEGsZV4uXdFcqXY0XAmXAn4GHgIf94ysTRXLahg0bGDBgAMOGDev0vdXV1Tz77LO6nFwkx6VbeI0dO5Z33nln70B7kXSk0tU4iVi34IvADOA1M1sUdmAiUdiwYQMVFRWkctvR9nFey5Z1NnewiPRU7t6tFi+AtWvXZjosySOpdDUeCVxJbDD8pcA7wFMhxyUSiTfffLPTbsZ2ms9LJPdt374dd0+7xQvQOC/pllS6Gm8BBgE/B8a7+3R3/364YYlEo73FKxUa5yWS+7pyg+x9HX744ZiZxnlJt3Q6gaq7nxNFICJR++ijj/jwww9TLrwAjjvuOB599FHcPaXuSRHpWbpTePXv35+DDz5YLV7SLal0NY42swfNbFV8Lq/1ZrY+iuBEwrRx40ag8ysaO5o4cSJbtmxh0yZdXyKSi7pTeIGubJTuS6WrcS7wK2A3MB24E7g7zKBEopDqVBIdVVVVAWgiVZEc1d3Cq30ur9jk5CJdl0rh1d/dnwTM3Te6+xzgc+GGJRK+dAqvz3zmM4AKL5FclYkWr8bGRt57771MhiV5JLDwMrM/mFklsNPM+gBrzexrZnYBkN4ZK9KDtM/hNXTo0JS3GTx4MIceeqimlBDJUZlo8QJd2SjpS9biNRd4HHgMGAh8A5gMXAVcE35oIuHqyhxeHVVVVanFSyRHZaLFC9A4L0lbYOHl7g8Ak4i1bi0CLgNWAM8Bx0cSnUiI3nzzTSorK7u83cSJE1m3bh2NjY0hRCUiYepu4XXwwQdTVFSkFi9JW2djvHYBzUARUEqsCCuJPxfJaV2Zw6uj9gH2r776amYDEpHQdbfw6tOnD+PGjWPJkiWZDEvySLIxXmcCy4EBwCR3v9ndf9D+iCpAkTBs27aNbdu2davwUnejSO5pamrCzOjfv3/a+7jwwgt5+umn916gI9IVyVq8bgK+4O43uvv2qAISiUI6c3i1GzlyJGVlZSq8RHJQ+30auzMB8syZMwG48847MxWW5JFkY7xOdPeVUQYjEpV0ppJoZ2ZUVVXpykaRHNTU1ERpafdGyxxyyCGccsopzJs3jz179mQoMskXqczjlXFmdqaZrTGzdWZ2YzZikPzWncILYgPsV6xYwe7duzMXlIiErr3Fq7uuvfZa3nzzTZ599tkMRCX5JPLCy8wKgF8CZwETgMvNbELUcUh+27BhAwMHDqSsrCyt7auqqti5cydvv/12hiMTkTBlqvC64IILKC0tZd68ed0PSvJKNlq8pgLr3H29u+8CFgDnZSEOyWPpzuHVrn2A/bp16zIYlYiELVOF18CBA7nkkkt44IEH9l4pKZKKvlk45kigYzNBHTAtC3Hs9f/+3//jjjvuYP/9989mGD3W1q1be11uFi9ezPHHpz8d3dixYykqKmL+/PksXbo0g5H1Hr3xvMkU5SaxKPKybNmybn33O7r22mv59a9/zSmnnBJ63DpngnU1N//+7/++9w4E2ZCNwislZjYLmAVQXl5ObW1taMd67bXXaGho0F8tAfbs2dPrcjN8+HAmTZrUrfPqoosuYunSpbz11luZC6wX6Y3nTaYoN4lFkZcRI0Zw1FFHZeT/FHfn9NNP5+233w49bp0zwbqam+eee45NmzaFGFFyFvUd1s3sWGCOu58Rf/1dAHf/SdA2U6ZM8bAnq6utraW6ujrUY+Qq5SaYchNMuQmm3CSmvARTboL1xNyY2VJ3n5JoXTbGeL0EjDazSjPrR+xWRI9kIQ4RERGRSEXe4gVgZmcDtwIFwB3u/uNO3r8F2BhyWEOBD0I+Rq5SboIpN8GUm2DKTWLKSzDlJlhPzM0h7j4s0YqsFF49kZktCWoWzHfKTTDlJphyE0y5SUx5CabcBMu13GRlAlURERGRfKTCS0RERCQiKrw+dnu2A+jBlJtgyk0w5SaYcpOY8hJMuQmWU7nRGC8RERGRiKjFS0RERCQiKrxEREREIqLCS0RERCQiKrxEREREIqLCS0RERCQiKrxEREREIqLCS0RERCQifbMdQCqGDh3qFRUVoR6jubmZgQMHhnqMXKXcBFNugik3wZSbxJSXYMpNsJ6Ym6VLl34QdJPsnCi8KioqWLJkSajHqK2tpbq6OtRj5CrlJphyE0y5CabcJKa8BFNugvXE3JjZxqB16moUERERiYgKLxEREZGIqPASERERiUhOjPESERGR3qe1tZW6ujpaWlrS3sfgwYNZvXp1BqNKXXFxMaNGjaKwsDDlbVR4iYiISFbU1dVRWlpKRUUFZpbWPhobGyktLc1wZJ1zd+rr66mrq6OysjLl7dTVKCIiIlnR0tJCWVlZ2kVXNpkZZWVlXW6tU+ElIiJ56e/+7u/405/+lO0w8l4uFl3t0ok9pa5GMxsCjAB2ABvcfU+XjyQiItJD7N69m1/96lf86U9/4vXXX6dvX428yUf19fWceuqpAGzevJmCggKGDRtGY2MjBx98MO+99x5mxqxZs7j++uszcszAM83MBgNfBS4H+gFbgGKg3MxeAG5z94UZiUJERCRCjY2NAPz1r3/l7rvvpqamJrsBSVaUlZWxfPlyAObMmUNJSQk33HADmzZtYtOmTUyaNInGxkYmT57MjBkzmDBhQrePmayr8UHgbeBEdx/r7ie4+xR3Pwi4BTjPzK4L2tjMDjKzhWa2ysxWmtn18eX7m9kTZrY2/nNItz+FiIhIF7QXXgD/+I//yO7du7MYjfQ0w4cPZ9KkSQCUlpYyfvx43nnnnYzsO7DwcvcZ7n6Xu29LsG6pu892918n2fdu4JvuPgE4BviqmU0AbgSedPfRwJPx1yIiIpFpL7yuuuoq/vrXv3LXXXdlOSLpqTZs2MCyZcuYNm1aRvbXaae2mU1KsPgjYKO7B/6J4O6bgE3x541mthoYCZwHVMffNh+oBb7TpahFRES6oaGhAYDLL7+c1atX84//+I9cddVVXZqPSTJr9uzZe7v9uqKtrY2CgoKE66qqqrj11lvTjqmpqYmLLrqIW2+9lUGDBqW9n45SuarxNuAF4Hbgv4DngQeANWZ2eioHMbMKYCKwGCiPF2UAm4HyLsYsIiLSLe0tXoMGDWLOnDmsX7+e+++/P8tRSU/S2trKRRddxJVXXsmFF16Ysf2mchnHu8B17r4SIN5d+EPg28BDQNJrcc2sBPgNMNvdGzpeeunubmYesN0sYBZAeXk5tbW1KYSavqamptCPkauUm2DKTTDlJphyk1iUeXn++ecBWLNmDZWVlQwfPpyf/exnjBw5MpLjd1VvPWcGDx68twj+0Y9+lNY+krV4wSfH8yWzc+dOCgsLaWxsxN358pe/zGGHHcaXvvSlpPtoaWnp0r9NKoXXmPaiC8DdV5nZOHdf39n8FWZWSKzousfdH4ovfs/Mhrv7JjMbDryfaFt3v51YKxtTpkzx6urqFEJNX21tLWEfI1cpN8GUm2DKTTDlJrEo87JhwwYATjnlFCorK/nbv/1b5syZQ2VlJYccckgkMXRFbz1nVq9e3e1Z5zM1c31RURFFRUWUlpayaNEiFixYwFFHHcWJJ54IwD/90z9x9tlnf2q74uJiJk6cmPJxUim8VprZr4AF8deXAqvMrAhoDdrIYlXZr4HV7v6zDqseAWYSuzJyJvBwytGKiIhkQPsYr/b/sK+55hpuvvlm7rzzTv7hH/4hm6FJlsyZM2fv8xNOOAH3hB1y3ZbKGK8aYB0wO/5YH1/WCkxPst3xwNXAKWa2PP44m1jBNcPM1gKnxV+LiIhEpr3rqL3wqqioYPr06cybNy+0/3BFIIUWL3ffYWa3AY+6+5p9Vjcl2W4RENQXeWrqIYqIiGRWY2Mj/fr1o6ioaO+ya6+9lmuuuYZFixbt7V4SybROW7zM7FxgOfDH+OsqM3sk5LhERERC09DQ8KlxQRdeeCElJSXMmzcvO0FJXkhljNfNwFRi823h7svNrDLMoESidMstt7B+/XoASkpKuPHGGznggAOyHJWIhCnRgOyBAwdyySWXcP/99/Pzn/+cgQMHZim6/OLuOXuj7HS6pVMpvFrd/aN9kqIOcOkVPvroI7773e8yaNAgBg4cyJYtW1i+fDlPPPFE0suTRSS3NTY2JpwQ84orruCOO+5g4cKFnHPOOVmILL8UFxdTX19PWVlZzhVf7k59fT3FxcVd2i7VqxqvAArMbDTwDeC5NGIU6XG2bt0KwM9//nNmzpzJvHnzuPbaa/nhD3/ID37wgyxHJyJhCZqC4Pjjj6eoqIja2loVXhEYNWoUdXV1bNmyJe19tLS0dLn4yZTi4mJGjRrVpW1SKby+DtwE7ATuAx4H0pvlTKSHaS+8hgyJ3au9pqaG2tpafvSjH3HCCScwY8aMbIYnIiFpaGhg6NChn1peXFzMMccc0ysnK+2JCgsLqazs3uil2traLs2jlW2dDq539+3ufpO7H+3uU+LPW6IITiRsH374IQD777//3mW//OUvGT9+PFdeeSXbtm3LUmQiEqagrkaA6upqli1bpu+/hCJp4WVmM83sZTNrjj+WmNk1UQUnErZ9W7wgNsD2jjvuYMuWLfzP//xPtkITkRAlm+28urqaPXv28Oyzz0YcleSDwMLLzGYSmzD1m8AIYCSx+zNeb2ZXRxKdSMgStXgBTJ06lSOPPFKXlYv0Uommk2h3zDHH7B3nJZJpyVq8vgJc4O4L3f0jd9/m7k8BFwFfjSY8kXAlavECMDNqamp44YUXWL16dTZCE5GQ7Nmzh6ampsCuxuLiYo499lgVXhKKZIXXIHffsO/C+LLEZ6tIjvnwww/p379/witirrzySgoKCpg/f34WIhORsDQ3NwMkvbGyxnlJWJIVXjvSXCeSM7Zu3fqp1q52Bx54IGeffTZ33XUXbW1tEUcmImHZ9z6NiVRXV+PuPPPMM1GFJXkiWeE13sxeTfB4DRgXVYAiYdq6deunxnd1VFNTw7vvvsuf/vSnCKMSkTA1NDQAyQuvadOmaZyXhCLZPF7jI4tCJEs+/PDDwBYvgHPOOYeysjLmzZvHWWedFWFkIhKW9havoDFeEBvnddxxx6nwkoxL1uL1lrtvDHoAWK7N7y+yj85avPr168fll1/Oww8/zI4d6mEX6Q1S6WoEOOWUU1i+fDnvvPNOFGFJnkhWeC00s6+b2cEdF5pZPzM7xczmAzPDDU8kXJ21eAGceeaZ7Ny5kxdeeCGiqEQkTKl0NQJcdtlluDt33313FGFJnkhWeJ0JtAH3mdm7ZrbKzNYDa4HLgVvdfV7QxmZ2h5m9b2YrOizb38yeMLO18Z/J/8cTCVlnLV4AJ5xwAn369FGXg0gvkUpXI8Dhhx/OiSeeyLx583D3KEKTPBBYeLl7i7vf5u7HA4cApwKT3P0Qd/+Suy/rZN/ziBVvHd0IPOnuo4En469FsmLXrl00Nzd32uI1ePBgJk2apMJLpJdItasRYhfYvP766yxevDjssCRPdHqvRgB3b3X3Te6+LdUdu/szwNZ9Fp8HtE+KNB84P9X9iWRa0Kz1iVRXV/PCCy9onJdIL9CVwusLX/gCAwYM0F0sJGNSKrwyqNzdN8WfbwbKIz6+yF7thVdnLV4QK7x27drF888/H3ZYIhKyhoYG+vTpQ//+/Tt9b2lpKRdddBELFizQH16SEcmmkwiVu7uZBXaam9ksYBZAeXl56N08TU1N6koK0Ftzs2JFbPjh22+/ndLn69OnD/PmzaNPn4//XumtuckE5SaYcpNYVHlZvXo1AwYM4Omnn07p/VVVVdx111385Cc/4ZRTTgk5usR0zgTLudy4e9IHcB0wurP3BWxbAazo8HoNMDz+fDiwJpX9TJ482cO2cOHC0I+Rq3prbh599FEHfPHixSm9/+ijj/YTTzzxE8t6a24yQbkJptwkFlVeampq/KCDDkr5/W1tbX7IIYf4GWecEWJUyemcCdYTcwMs8YCaJpWuxoOB/zSz9Wb2QHyKiao067xH+HgKipnAw2nuR6Tb2m+QncoYL4h1Ny5evJjt27eHGZaIhKyhoSGl8V3t+vTpw3nnnceiRYt0+zDptk4LL3e/2d1PAY4AngW+BSztbDszuw94HhhrZnVmdh1wCzDDzNYCp8Vfi2RFV8Z4wcfjvDSfl0hua2xs7HQqiX1NnDiR5uZm/vrXv4YUleSLTsd4mdn3gOOBEmAZcAOxAiwpd788YNWpXQlQJCztLV777bdfSu9vn89r4cKFWRvnISLd19jY2KUWL4iN8wJYtmwZY8aMCSEqyRepdDVeCJQBfwYeAh72j69MFMlZH374IYMHD6agoCCl9w8aNIjJkyfn1iBOEfmUdAqvCRMmUFhYyPLly8MJSvJGKl2Nk4h1C74IzABeM7NFYQcmErZUZq3f1/Tp0zXOSyTHdXWMF8Tu2zphwgQVXtJtnRZeZnYkcCWxwfCXAu8AT4Ucl0joUrlP476qq6tpbW3VfF4iOSydMV4QG+elwku6K5WuxluAQcDPgfHuPt3dvx9uWCLhS6fF6/jjj6egoEDdjSI5yt3T6mqE2DivzZs3s3nz5hAik3yRSlfjOe7+z+7+nLu3RhGUSBS2bt3a5RYvjfMSyW0tLS20tbWlXXgBavWSbkmlq3G0mT1oZqvic3mtN7P1UQQnEqYPP/ywyy1eoPm8RHJZQ0MDQFpdjZ/97GcBFV7SPal0Nc4FfgXsBqYDdwJ3hxmUSNjcPa0WL4gNsG9tbeW5554LITIRCVNXbpC9r/3224+KigoVXtItqRRe/d39ScDcfaO7zwE+F25YIuFqbm5m9+7dabV4aZyXSO7qTuEFGmAv3RdYeJnZH8ysEthpZn2AtWb2NTO7gNhkqiI5q33y1HRavEpLS5kyZYoKL5Ec1N7VmG7hVVVVxRtvvEFzc3Mmw5I8kqzFay7wOPAYMBD4BjAZuAq4JvzQRMLTfrugdFq8IDbO68UXX2THjh2ZDEtEQtbe4pXOGC+IFV7uzquvvprJsCSPBBZe7v4AMIlY69Yi4DJgBfAcsVsIieSs7rR4wcfzea1cuTKTYYlIyLrb1agrG6W7OrtX4y6gGSgCSoE9oUckEoHutnidcMIJFBQUsGzZskyGJSIh627hddBBBzFkyBB99yVtgYWXmZ0J/Ax4BJjk7rp2XnqN7rZ4lZSUcNJJJ/H000/j7phZJsMTkZB0ZzoJADPjxBNP5NFHH2X37t307dtZ+4XIJyUb43UT8AV3v1FFl/Q23W3xApg5cybvvPOOppUQySHtLV4DBw5Mex8zZ85k06ZNPPHEE5kKS/JIsjFeJ7q7BrBIr7R161b69u3brV++F110EcXFxcybNy9zgYlIqBobGykpKaFPn1RmU0rsnHPOoaysTN99SUv6Z55IDmuftb47XYQlJSVUV1fzP//zP7q0XCRHNDQ0pD2+q12/fv244oor+N3vfrd32IJIqrJSeJnZmWa2xszWmdmN2YhB8lu6s9bv64wzzqCxsZHf/va3GYhKRMLW2NiY9viujq699lp27drFggULMhCV5JPICy8zKwB+CZwFTAAuN7MJUcch+S3d+zTu6zOf+QyHHnqouhxEckRjY2O3W7wgNq3EZz7zGX33pcuycTnGVGCdu68HMLMFwHnAqizEAsCmTZt48803GTZsWLZC6NF6Y242bdrEIYcc0u399OnTh5kzZzJnzhyefvpphg4dmoHoeofeeN5kinKTWBR5ef/99zPS4mVm1NTU8H/+z//hscce4+CDD85AdMF0zgTram4qKysZMGBAiBF1wt0jfQAXA//d4fXVwC+SbTN58mQP0/XXX++AHnn2+OIXv9jtc2fhwoX+5ptvupll/fPooYceqT0uueSSDPzP4f7ee+95YWFh1j+PHl17LF68OCP//skASzygprHY+uiY2cXAme7+N/HXVwPT3P1r+7xvFjALoLy8fHKY/ejr1q1j3bp1FBcXh3aMXNbS0tIrc/PZz3622+O8mpqaKCkpYeXKlWzZsiVDkfUOvfW8yQTlJrGo8nLkkUdmrHX69ddfZ/PmzRnZVzI6Z4J1NTeTJk3KSKtnMtOnT1/q7lMSrctG4XUsMMfdz4i//i6Au/8kaJspU6b4kiVLQo2rtraW6urqUI+Rq5SbYMpNMOUmmHKTmPISTLkJ1hNzY2aBhVc2rmp8CRhtZpVm1o/YPSAfyUIcIiIiIpGKvMULwMzOBm4FCoA73P3Hnbx/C7Ax5LCGAh+EfIxcpdwEU26CKTfBlJvElJdgyk2wnpibQ9w94Yj/rBRePZGZLQlqFsx3yk0w5SaYchNMuUlMeQmm3ATLtdxo5noRERGRiKjwEhEREYmICq+P3Z7tAHow5SaYchNMuQmm3CSmvARTboLlVG40xktEREQkImrxEhEREYmICi8RERGRiKjwEhEREYmICi8RERGRiKjwEhEREYmICi8RERGRiKjwEhEREYlI32wHkIqhQ4d6RUVFqMdobm5m4MCBoR4jVyk3wZSbYMpNMOUmMeUlmHITrCfmZunSpR8E3SQ7JwqviooKlixZEuoxamtrqa6uDvUYuUq5CabcBFNugik3iSkvwZSbYD0xN2a2MWiduhpFREREIqLCS0RERCQiKrxEREREIpITY7xERESkd2ptbaWuro6Wlpa0th88eDCrV6/OcFSpKS4uZtSoURQWFqa8jQovERERyZq6ujpKS0upqKjAzLq8fWNjI6WlpSFElpy7U19fT11dHZWVlSlvp65GERERyZqWlhbKysrSKrqyycwoKyvrckudCi8RERHJqlwrutqlE3dKXY1mNgQYAewANrj7ni4fSaSHeumllzjyyCPp379/tkMRkQgtW7aMww8/PCvdVNIz1NfXc+qppwKwefNmCgoKGDZsGC0tLQwYMIC2tjZ2797NxRdfzA9+8IOMHDOw8DKzwcBXgcuBfsAWoBgoN7MXgNvcfWFGohDJkh07dnD88cfz9a9/nX/7t3/LdjgiEhF354QTTuDKK6/k9ttvz3Y4kiVlZWUsX74cgDlz5lBSUsINN9yAu9Pc3ExJSQmtra2ccMIJnHXWWRxzzDHdPmayrsYHgbeBE919rLuf4O5T3P0g4BbgPDO7LmhjMzvIzBaa2SozW2lm18eX729mT5jZ2vjPId3+FCJp2rZtG62trdx11120trZmOxwRicj27dvZvn07CxYsYPv27dkOR3oYM6OkpASIXXXZ2tqase7QwBYvd5+RZN1SYGkn+94NfNPdXzazUmCpmT0B1ABPuvstZnYjcCPwnS5HLpIBDQ0NAGzZsoXHHnuMc889N8sRiUgU2r/7jY2N/Pa3v+XKK6/MckQCMHv27L0tUKlqa2ujoKAgcH1VVRW33nprl2Npa2tj8uTJrFu3jq9+9atMmzaty/tIpNPB9WY2KcHjMDNLOj7M3Te5+8vx543AamAkcB4wP/62+cD53foEIt3Q/ssXYO7cuVmMRESipO++dKagoIDly5dTV1fHiy++yIoVKzKy31QG198GTAJeBQw4ElgJDDazr7j7nzrbgZlVABOBxUC5u2+Kr9oMlKcRt0hGtP/ynTJlCo8++ihb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y81sQtRxSH7LRFcjwOTJkxkxYoS6HERyRLr3adzXcccdx+jRo/Xdly7rm4VjTgXWuft6ADNbAJwHrMpCLABs376dxsbGvX3/8km9MTfbtm3j0EMP7fZ+CgoKuPrqq/npT3/K2rVrGTp0aAai6x1643mTKcpNYlHkZcuWLd0a39XOzKipqeGmm27ilVde4eCDD85AdMF0zgTram5KS0vp2zcb5U+cu0f6AC4G/rvD66uBXyTbZvLkyR6m66+/3gE98uwxc+bMbp87Cxcu9NWrV2f9s+ihhx6pP84777xuf/fd3d966y03s6x/Hj269li8eHFG/v2TAZZ4QE2TxZIvOTObBcwCKC8vD/WmpBUVFXzpS1+iX79+oR0jl+3atatX5ubYY4/t9nnV1NTE5s2b+eEPf8h7772XmcB6id563mSCcpNYVHmZOnVqxv5P+fGPf8w777yTkX0lo3MmWFdzU1dXx/bt2ZuswTziyR/N7FhgjrufEX/9XQB3/0nQNlOmTPElS5aEGldtbS3V1dWhHiNXKTfBlJtgyk0w5SYx5SWYchOsJ+bGzJa6+5RE67JxVeNLwGgzqzSzfsRuRfRIFuIQERERiVTkLV4AZnY2cCtQANzh7j/u5P1bgI0hhzUU+CDkY+Qq5SaYchNMuQmm3CSmvARTboL1xNwc4u7DEq3ISuHVE5nZkqBmwXyn3ARTboIpN8GUm8SUl2DKTbBcy01WJlAVERERyUcqvEREREQiosLrY7dnO4AeTLkJptwEU26CKTeJKS/BlJtgOZUbjfESERERiYhavEREREQiosJLREREJCIqvEREREQiosJLREREJCIqvEREREQiosJLREREJCIqvEREREQi0jfbAaRi6NChXlFREeoxmpubGThwYKjHyFXKTTDlJphyE0y5SUx5CabcBOuJuVm6dOkHQTfJzonCq6KigiVLloR6jNraWqqrq0M9Rq5SboIpN8GUm2DKTWLKSzDlJlhPzI2ZbQxap65GyWvuzhe/+EWeffbZbIciIiJ5ICdavETC8tFHHzF37lyWLl3KsmXL6NNHf4uIiEh49L+M5LWmpiYAXn31VX73u99lNxgREen11OIlea298AKYM2cO559/vlq9REQi0traSl1dHS0tLWnvY/DgwaxevTqDUaWuuLiYUaNGUVhYmPI2Krwkr7UXXpdddhkLFizg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xIm8+OKL2Q5HRER6ORVektc++ugjGhoaWL16Ncceeyw33ngjbW1t2Q5LRER6KRVekteampoAuPXWW5k5cyb//M//zO9+97vsBiUiIr1WTozxEglLe+E1cuRI/vZv/5Z7772X5557josuuijLkYmI9H6tra3U1dXR0tKS9j4GDx7M6tWrMxhV6oqLixk1ahSFhYUpb6PCS/Jae+FVUlJCv379mDRpEosXL85yVCIi+aGuro7S0lIqKiows7T20djYSGlpaYYj65y7U19fT11dHZWVlSlvp65GyWsdCy+AadOmsXTpUlpbW7MZlohIXmhpaaGsrCztoiubzIyysrIut9ap8JK8lqjwamlp4bXXXstmWCIieSMXi6526cSeUuFlZkPM7AgzO9TMVKxJr5Go8ALU3Sgikgfq6+upqqqiqqqKAw88kJEjR+59/f7773PxxRczbtw4xo8fz/PPP5+RYwaO8TKzwcBXgcuBfsAWoBgoN7MXgNvcfWFGohDJkn0Lr4qKCg444AAWL17MV77ylWyGJiIiISsrK2P58uUAzJkzh5KSEm644QYAZs6cyZlnnsmDDz7Irl272L59e0aOmWxw/YPAncCJ7r6t4wozmwxcbWaHuvuvE21sZgfFty8HHLjd3f/dzPYH/geoADYAl7j7h938HCJp2bfwMjOmTZumFi8RkTz20Ucf8cwzzzBv3jwA+vXrl9bdTRIJLLzcfUaSdUuBpZ3sezfwTXd/2cxKgaVm9gRQAzzp7reY2Y3AjcB3uhy5SAY0NTVhZvTv33/vsmnTpvG///u/bNu2jf322y97wYmI5JHZs2fvbX3qira2NgoKChKuq6qq4tZbb+3yPt98802GDRvGtddeyyuvvMLkyZP593//94zcE7LT8VpmNinB4zAzSzoVhbtvcveX488bgdXASOA8YH78bfOB87v1CUS6oampiYEDB9Knz8dfhfZxXi+99FK2whIRkSzavXs3L7/8Ml/5yldYtmwZAwcO5JZbbsnIvlOZx+s2YBLwKmDAkcBKYLCZfcXd/9TZDsysApgILAbK3X1TfNVmYl2RIlnR1NS0t5ux3dFHH42ZsXjxYmbMCGz4FRGRDEqnZQrCmcdr1KhRjBo1au8f4hdffHGkhde7wHXuvhLAzCYAPwS+DTwEJC28zKwE+A0w290bOl566e5uZh6w3SxgFkB5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"
class="
jp-needs-light-background
"
>
</div>

</div>

</div>

</div>

</div>
</body>







</html>
