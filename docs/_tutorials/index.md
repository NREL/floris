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
  
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
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
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="Initialize-FlorisInterface">Initialize FlorisInterface<a class="anchor-link" href="#Initialize-FlorisInterface">&#182;</a></h2><p>The <code>FlorisInterface</code> provides functionality to build a wind farm representation and drive the simulation. This object is created (instantiated) by passing the path to a FLORIS input file. Once this object is created, it can immediately be used to inspect the data.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
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
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="Build-the-model">Build the model<a class="anchor-link" href="#Build-the-model">&#182;</a></h2><p>At this point, FLORIS has been initialized with the data defined in the input file. However, it is often simpler to define a basic configuration in the input file as a starting point and then make modifications in the Python script.
  This allows for generating data algorithmically or loading data from a data file. Modifications to the wind farm representation are handled through the <code>FlorisInterface.reinitialize()</code> function with keyword arguments. Another way to
  think of this function is that it changes the value of inputs specified in the input file.</p>
  <p>Let's change the location of turbines in the wind farm.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
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
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <p>Additionally, we can change the wind speeds and wind directions. These are given as lists of wind speeds and wind directions that will be
  expanded so that a wake calculation will happen for every wind direction with each speed.</p>
  <p>Notice that we can give <code>FlorisInterface.reinitialize()</code> multiple keyword arguments at once.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
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
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
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
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>
  
  
  
  
  <div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
  <pre>(0.0, 150.0)</pre>
  </div>
  
  </div>
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  
  
  <div class="jp-RenderedImage jp-OutputArea-output ">
  <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPoAAADyCAYAAABkv9hQAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABnaUlEQVR4nO29d3hc5Zk2fp/pGo2k0Yx6r7Ylq7tRjNkYCCTgQrNN8gNC2SQkLKSwgSRsQnaXhLAsCbubj2S/EELyJaHYtNiGhSUQCBgwtmz13seSZjRFml7f3x/Sezgzmt4kWXNfFxfWSDrnjObc532e572f+2EIIUghhRTOb/BW+gJSSCGFxCNF9BRSWAdIET2FFNYBUkRPIYV1gBTRU0hhHSBF9BRSWAcQhPh+au8thRQSDybRJ0it6CmksA6QInoKKawDpIieQgrrACmip5DCOkCK6CmksA6QInoKKawDpIieQgrrACmip5DCOkCK6CmksA6QInoKKawDpIieQgrrACmip5DCOkCK6CmksA6QInoKKawDpIieQgrrACmirwAIIXA4HHC5XEjZbaeQDIQynkghzvB4PHA4HLDZbOxrfD4fQqEQAoEAfD4fDJNwH4IU1hmYECtKarmJEwghcLlccLlcsNlsGBkZgUwmg1wuh1gsBiGEJbjdbkdGRgZEIlGK+OsDCf+AU0RPAmio7vF4oFarMTw8jPLyctjtduj1ethsNshkMmRnZ0Mul2NoaAgVFRWQSqUAUiv+OkCK6GsdLpcLTqcTbrcbAwMDcDgc2Lx5s9cKTgiB0WiEwWCAXq/H/Pw8srOzkZubC7lcDpFIBI/Hw/68QCBg/0sR/7xAiuhrFdxQ3Ww2o6urC8XFxSgtLQUAOByOgATt7OxEXl4erFYr9Ho9nE4nMjMz2RVfKBR6FfEEAgG74vN4vBTx1x4S/oGlinEJgMfjYVfxc+fOYXJyEo2NjcjIyACAkJV2Ho8HmUyG/Px8VFRUwOPxYGFhAXq9HufOnYPL5UJWVhbkcjnkcjkYhoHL5QIAMAzjteKniJ8CkCJ6XEEIgdvtxvDwMKRSKc6dOweBQIDt27dDIFj+p+aG78HA4/FYUgOA2+1miT81NQW3281+PysrCwDgdDoBpIifwiJSRI8TCCHsKr6wsIDx8XFs2LABhYWFy36WYRgwDBNwZQ/2PWCxOJednY3s7GwAi8Sfn5+HXq/HxMQECCEp4qfghRTR4wC6N+7xeDA2NgadTofq6mq/JE8E+Hw+FAoFFAoFgMUCICX+2NgYGIaBXC5HdnY2MjMz4XQ6odVqYTKZUFxczOb4fD4/RfzzFCmixwBuwc3hcKCrqwsZGRkoKSmBUCiM+rihVvRQEAgEUCqVUCqVABZX8/n5eWi1WoyMjIDH40EsFgMACgsL4XQ6vVZ8WtgTCARs9JHC2kaK6FGCuzeu1WoxMDCAjRs3IicnByMjIzFLW+MpjRUKhcjJyUFOTg6AReJPTExAp9Ohvb3dKxXIyMiAw+GA3W4HsFgfEAqF7IqfIv7aRIroUcDtdrP5+NDQEEwmE7Zu3cqukrGuyIkmklAoRFZWFhiGQVVVFRwOB/R6PWZnZzE4OAiBQMASXyaTscRnGAY8Hm9ZqJ/C6keK6BGAG6pbrVZ0dXUhNzcXW7Zs8SJnPIiezGYXkUiE/Px85OfnAwCr2Dt37hyMRiPEYjFb3KMrvsPhAIAU8dcIUkQPE3Rv3OPxYGZmBqOjo9i8eTO75cVFsokab4jFYhQUFKCgoAAAYLPZoNfroVKpYDKZIJFI2OJeenp6ivhrACmihwDdG6ehel9fH9xuN7Zv3x6w4LYWVvRIji+RSFBYWIjCwkIQQljiT0xMwGw2Iy0tjVXtpaWleRHf7XaDz+cjPT09RfwVRIroQcDdGzeZTOjq6kJZWRmKi4uD5tE8Hg8ejyfosU0mE4RCIZvXrwSiqQUwDIO0tDSkpaWhqKgIhBBWqjs2Ngaz2Yz09HSW+AsLC7DZbCgrKwPwaXGP6vRTxE8OUkQPALo3/sEHH6CkpATnzp1DU1MTZDJZyN9lGCYg0T0eD/r7+7GwsACPxwNCCLKyslhi0C2ttRL6MwwDqVQKqVSK4uJiEEJgNpthMBgwMjKChYUF9oFGW3LtdjvsdjsIIV5hPn3vKcQfKaL7gBuq06KbyWTC9u3bwefzwzpGIKJaLBa2YaWqqgrAIvFp1xoVtxBCIBaLkZWVteZWPIZhIJPJIJPJ2AekxWJhdyh8W3J5PF7KhCMJSBGdA+7euMFgQG9vL4RCITZv3hzRcfwRfXZ2FkNDQ2wBjxb2+Hz+MnFLb28vDAYDZmZmIBQKoVAo2D3ueN34yYwYJBIJSkpKUFpa6tWSOzAwwJpsUOIzDJMifgKQIvoSuDLW0dFRaLVatLW1ob29PeJjcYlOQ3Wr1Ypt27ZBJBIF/V2hUAipVMpKWu12O3Q6HaampmA0GtnCl0KhgFQqjenGTwZpfBt3GIZBZmYmMjMzUVZWBo/HA6PRCL1ej97e3mUtuQzDwGq1ssdIET86rHuic/fG7XY7urq6IJfLsW3btqjDZkp0bqi+adOmsG9K7oNCLBZ7Vbxp4Wt0dBRms5kNg7Ozs5GWlhbV9SYShJCgf0cej4esrCxkZWWF3ZKbIn7kWNdE5+6Nz83NYXBwEJs2bWLD6GjBMAxMJhPa29sD7rVTON0eaM1OKKRCiATBHyz+Cl8mkwl6vZ4Ng+lqmJ2dHTJ6SAbCbcWliLQll0v8mZkZFBcXp4jvB+uS6NyCm8fjweDgICwWS1ihdSh4PB5MTU3BZDLhggsuCHo8QgiOdamhMtiQlyHCdS2F4POCt7BywTAMMjIykJGR4RUG63Q6qFQqlhTcij733MlApET3RbgtudnZ2ZiamkJhYaHXip9y31nEuiM6N1S3WCzo6upCQUFBRKF1INBQXSaTQSqVhnxouDwE5+ZtyE4XQm1ywO7yQCoKr7LvD9wwuLKykiWFTqdb1q7K9aBLJDweT1x3DoK15FqtVrS3t3u15LrdbtY/n/bir0firyuic0P16elpjI+PY/Pmzaw5QyzgVtWp26s/uD0EJrsLEj6BkM/DrhoF2icXcGFFNkvyeO2j+5LC6XTCYDBgbm4Oc3Nz7HloRT8RW3mxruihwG3J1ev1aGpqWtaSS4mfkZHBEh9YXyYc64LovjLW3t5eAAho8eTv9wPdAP6q6nq93i9R3R6CF9tVGNdZsSFXiivrctBQlImGokyvn0uUYEYoFCI3Nxe5ubnIzMyE2WyGRCJhm1ckEgkbJqenp8flpk800X3P5a8lV6/XQ6PRYGhoaFlLLnXpBc5v4p/3RKcy1tOnT6Oqqgrd3d0oLy9HcXFxWL9P5az+xDKBquoBBTMON8Z1VhRkiNE/a8JlG5UQ8lfuRhIKhWzzCtWw0zCfK2VVKBRRV/STSXR/EAqFyMvLQ15eHgCEbMn1NeE4X9x3zmuic/fGDQYDenp60NzcjPT09LCPEYi0vgIY39/hSmA9HgIej4FMzEdTcRY6VQvYUSGHgOf/plkJCSzVsBcXF3tJWfV6PQYHB2Gz2VhhS3Z2dtga/WQSPZzzhNuSG4j4vjr9tUL885Lo3IKb0+lEd3c3CCHYsWNHxHmob4NKOAIYSlSn24OX2qcxqjXjyvo8tJTK8dn6PFy+KRdutytk48tKgitlLS0t9RK29PT0sPvbXK95f0gW0aN9MAZqyaU7J/5acu12O86dO4f8/HxIpdI1Ybt13hGdK2PV6/Xo6+tDbW0tbDZbVMUmLtHDFcBQomuMDgxpTFCmi/D+sA4tpfKlYzJwuUhAEqy2NlVgubDF3zYXXe2zsrLYVCdZRI/XLkK4LblqtRp5eXle7jt0xV+NvfjnFdHpCk5lrHq9Hlu2bIFEIsHQ0FBUNx0lXbBQPdDvKNKFyMsQQ2Ny4OJqBfv9ubk5tiAolUq98uBkrgix7m/7bnPRiv7w8DBb9OLq1hOJUAq8aBCsJddqteLs2bNeLbncXvy7774bDz74IDZt2hTOeX4D4BoAakJIw9JrDwH4ewCapR/7HiHk+NL3vgvgDgBuAPcQQv4n1DnOC6L7Tirt6uqCQqHAtm3b2Js5WFEtGBiGwdDQEJxOZ9iCGkp0iZCPWy8sg8XhRlba4hilkZERVkfP5/PZG4d2dmVkZIAQwk51WSsQCARe1W5u0au3t5d9oNHcN94PtHjv1/sDV5l47tw5bNmyBRaLhW3JtVgskMlkGB4ehk6ng0QiCffQvwXwXwB+5/P6zwghj/lcQz2AQwA2AygC8L8Mw2wghLiDnWDNE527N063UOrq6tiVhiIaolssFuh0OpSUlKChoSEqrbqQz0NWGg8Oh4MV02zduhUejwdutxvp6elIT09HSUkJmwePjY1BpVJhdnbWS9kW6UNqJUGLXvTvJxAI2BDYZDKxKyHV6MdK/GQQ3Rd0dBZtyaWS5HfeeQf9/f3Ys2cPtmzZgp///OdBo0BCyLsMw1SEedp9AJ4lhNgBjDIMMwRgO4ATwX5pzRLdd2+cVoYDrbrhuL5wQUN1uVyOwsLCiPXa3Bx4fn4eXV1dqK2tZbd5qOkE97g0D1YqleDxeMjLy2OVbSMjI17hcqwtq8mWwPqGwPQh6tujrlAoonLdSTbR/f39qCT561//Oo4cOYK//vWv6O3tjSU6u5thmFsAfALg24QQPYBiAB9yfmZq6bWgWJNE51o80UmlRUVFqKurC3jzh0t036r64OBgxKSgKzohBJOTk1CpVGhtbWXnnYcL30EMdCtIpVJhYWEh5vx+JdpU6XlpJMPtUdfpdGxFn9ucE84wjJVY0YPB6XQiLS0NW7ZsifYQTwL4FyxONP4XAP8O4PZoD7bmiM7dG5+ensbExAQaGhqQmZkZ9PfCIbq/qnowW6hAYBgGbrcbHR0d4PP5EbnT0N/393DhbgXRwhB3VaT73AqFYlV0rgHhVd25Peq0VZVW9CcnJ70aVwKlMMkmeqIfkoSQWc65/i+Ao0tfqgCUcn60ZOm1oFgzRKf5j91uh0QiQU9PD3g8Xtgy1lBED1RVjzTkBxYfGPPz86ivrw9bgRcpuIUhbn6v1+vR1dXFdq4pFIoVze+j2eng8XheHWu0os9NYej3MzMz2c8oWUQPFeHFY0uRYZhCQsj00pfXAuha+verAP7IMMzjWCzG1QL4ONTx1gTR6d64Xq+HWq2G0WhERUUFioqKwj5GIMKGEsBEuqc9PT2NkZERtmc8GkQTRfjb5/YlB9eSKpk5eqwE9FfRp1ZbAwMDEIlEkEgkfuseiQC1sA6GSK6DYZg/Afg7ADkMw0wB+CGAv2MYpgWLofsYgK8sHbebYZjnAfQAcAH4eqiKO7AGiM7tG9doNJibm8OOHTsiznf9ET0cAUwkuX1fXx/sdju2bt0alQUVRTxuVF8vOofDAZ1Ox0o9AbCWVYncv08E8UQikZd+3WazYWJiAvPz8/j444+9tvJitdvyh1BEd7lcEUVQhJCb/Lz8VJCffxjAw2GfAKuY6Ny9cYfDge7ubggEAuTl5UVMcmA5YcMVwPhW0P3BarWio6MD+fn5qKurYwtxwUBvPqfbg1mjHRliAbLSPi06xXvFFYlEXvk9bVwZHh6G1WpNWH6fjL53KlMVCoWoqKiAxWKBXq9n97a5Gv0I9rYDwu12B41SaEPQasKqJDp3b1yn06G/vx8bNmyAWCzG+Ph4VMfk8/nweDwRmzWGCqM1Gg0GBgZQX1/P5pRA+ET9aMyA/lkzxAIe9jXlI0OSeF93hmEgEokgEolQXFwMj8cDk8kEnU4X9/w+mRJY2mTC1SbQ2o5Op0NfXx8cDoeXRj+ah1ooPQb18ltNWFVE97V4GhkZwfz8PCtjNZvNcLtDpiN+wePxYLFYMDw8jPz8/LAdZQKt6IQQDA0NwWAweE1SBSILvfVmJ9KEPNhdHticbmRIkvORcAnI4/G8qt40v6cmlLQ4RvfvI8m5k9nU4u+6uHZb5eXlrPkkddb1eDzLBmiEQqjQPbWiBwF3b9xms6GzsxM5OTnYunXrMhlrNDCbzdDpdGhpaQmpVefC34rucDjQ0dGBrKwsr+sLB4QQGKxOSAQMGAAXVmWjfXIeeRli5MhE7DmTUSwLdN3+8ntuK6dEImELe6Fy4GSv6KHgz3zSd4AGtznH3zFDEd1kMqVWdH+ge+O0eWRkZGRZKAws3oCRrujcEUgVFRURkRxY/nChbZobNmxAbm5uRMcCgA9GdOiYWkCmmMHn6pTIkYlxRV3kx0kmuD3c3P17bg5MFXu+oXAyiR7OauwLfwM06O7O4OCg3wEaoXJ0KvFdTVhRovuG6v39/XA4HAEnldI8O1xYLBa2SEaH/EUKuqITQjA+Po6ZmRm0tbVF7bgypDYjje9Gx8AUsi1TUKYLWZLQZo+VMJ4IF77791xVW3d3N1wuF5vfUy++1bSih4KvIw13gAbtT6euM4EeYrS5ZTVhxYjO7RunMtaSkhKUlJQElbGGu6L7VtVVKhXrFBIJ6Gyws2fPQiQSYfv27RHfUNwbolrmwhsdKuxsqMClm/Lhcbug0+nYZg+ZTMY6mCQS8XqQ+KrafPN7i8WC0dHRqPL7SJAowYy/ARq0Bfrjjz/2O0Ajkhz99ttvx9NPP62Gd4vqvwHYA8ABYBjAbYQQw1LjSy+A/qVf/5AQ8tVwzrMiROfKWFUqFVQqFRoaGkKK/8NZGQJV1aPN7+12OyYnJ7Fx48aIBDoUH4/q8NGYHvUFMhQSLdJddvzg4MVgGAZOpxNCgfeNZDKZMDk5Ca1WC71en1B1WyJWWt9Q+KOPPmLbOiPN7yNBMttUMzMzoVAoUFBQsGyAxvT0NE6cOBG2WOpLX/oSnn766avg3aL6JoDvEkJcDMP8FMB3Ady/9L1hQkhLpNeeVKJTHzKDwYCsrCz09PRAIBBErAUPBG6o7ltVj4boKpUKExMTyM/Pj4rkHg/Bh6N6KNJ4+J9P+nHztiK0tNSxeZ4vaIU4NzcXaWlpKCsr81K3CQSCZWH+agePx1uW39M9bqvVCplMxhI/llnxyZTAut1uiMVir4o+HaDR0dGBw4cP469//SteffVVPPLII7j88ssDHmvXrl0AoOO+Rgh5g/PlhwBuiPWak0Z0ujdusVgwPj4Ou92O6upq1qsrVoQSwERCdLfbjb6+PrhcLmzcuBELCwtRXROPx6BExuCD3jE0VhWjfkN1WOSkObrv6kjzRW6YT4kfC0mSBW5+Tw0ofbvWAk2WCYVkE93fuXg8HlpaWtDc3IyDBw9i3759UaWLPrgdwHOcrysZhmkHsADgQULIe+EcJOFE5xbcCCGYnp7G/Pw8LrzwwqgUbr4IVwATLtFpVFBYWIiysjJotdqo8llavCtwz+If925FTpYM/ACur+HCN1+kQhAuScIN81dDsS9Yfj82NhbR/n2iiL5gc+HZTxabww5tLUamRBC2YCbW+5thmO9jUc/+h6WXpgGUEUK0DMNsAfAywzCbCSEhV6KEEt1XxtrZ2Yn09HRkZmbGheTBQnVfhEN0uqXCjQqiCfmpFsBoNGJHgLQk2LWGU3X3FYL4NrGEE+avttA/nP37QCOjEyW1fXdQi1MT8wCAoiwJrmnMD0swE2vVnWGYL2HRR+4ysnQzLLnK2Jf+fYphmGEAG7BoTBEUCSM6V8aq1WoxMDCAjRs3Qi6X49SpU1Efl5JArVaHbdYIBCcsHbRoNBqXRQXhaN25sFgsOHv2LPh8PhoaGsL+PS6iuWHPtzAf8L9/Hyi/T4Q5JADkyETgLX0eVNAUiuixbq8xDHMVgO8AuJQQYuG8ngtARwhxMwxThcUW1ZFwjhl3ovuG6kNDQzAajaxMlH4/WjAMg56eHtjt9oimnwYiut1uR0dHB7Kzs7Flyxa/bijhruhU997Q0IDu7u6wficQYg2tfcN8qgzs7e2F0+lkZa+5ublrwosuVH5vNBrB5/ORm5sbcX7vC5XBhlmjHZsLM7CjQo5s6aKmY0Pe4pZZPAUzN910E7Do98ZtUf0uADGAN5fuR7qNtgvAPzMM4wTgAfBVQojO74F9EFeic2WsVqsVnZ2dyM/P9yJQLOGVxWKByWRCTk4O6uvrIzqWP7ENvfE3btzI9jr7IpzQnRCC4eFh6PV6r4dPtKqweAtmGM4whrKyMrjdbgwMDMBiseD06dPsWCKlUrlmqvm++X17ezsUCgUMBkPE+T0XaqMdP/6fIdicbmwrl+Orl5RjY7736hxOjh6uT9yf/vQn/OlPfyr0edlviyoh5AiAI2Ed2AdxIzohBHa7HYQQzMzMYGxsDPX19RFLTgOBVtUzMjJQXFwclWsJJSxt01Sr1WzDTCCEIp3T6URHRwcyMjK8dO/091Yjafh8PtuLnp+fz3rRTU5Owmg0Ij09nQ3z49HWmQwQQqBUKtlRSzS/n56eRn9/f9D8ngu9xQmHywOxgIdz8/496cMJ3VebXXfciE73hnt7e+HxeMK2eAoF36p6b29vVKE/VdU5nU50dXVBIpFg27ZtIZ/0wVZ0o9GIzs5OVFdXszcYRSyrcrIlsL5edDTM7+vrg9PpRFZWFpsLr9Yw37fq7jtjzZ8+35/rbG1eOq6sz8Wwxowb2/xrJ0JV+M/77rWuri7k5OREteL6A62qFxQUsFX1aBVuPN6it/rJkydRVVUV9v59oGLcuXPnMDY2hqamJr+Fl9WsVwcC1wD8hfnUcpqGxOFaTifz/YequvsOkOTOkXtrzIazWgaX1ihw68WVuKHVN5L2RqhIze12x2WRiyfiejWtra1hfbi0wBXsqRhIABNNBxuwSEyLxYKLLroo4mmqvkMWqYFBsKglVLXebrfDYDBAJM3AGZURaUI+mksyIeTzVrxNlQvf0UvUkmpqaipkmJ/s1CUCjzY2v88tLMETnZ3IkhH874AOteJ5yEQ8NoKh5pPhnme1PtzjSvRwb1BKVn9EDyWAibSDze12o6enB4QQSKXSiEMqLmFpc0teXl5QD3kg+N/CYDCgu7sbMpkMpyYGMGvjQ5wmRYawCBsKs/3+zmqBryVVoDBfLpeznXjJQLTnkQh5qM1Lx5DGgk3FCuzcUQu3y+llPikWi9mHWTj6j2S+73ARd6KHA0p031ZUf6G6LyLpYDObzejo6GC74k6cCDq1xi/oik7DvE2bNrF71eH8ni/oQIeWlpbFcbs5FnwwqIHVYsbs1ATmp4aQlpbGCo1WWwjIRagwn2EY2O12LCwsxDxZJlHgMQy+c0U1VAYbCrMk4PMY8H3MJ2l+Pzo6CrPZzDavBNIkRLKqB+heU2BR9lqBRQfYA4QQPbP4B3wCwOcBWAB8iRByOpzzrMhd5C/8DtesMdzQnR6voaGB7YuOBgzDwGq1YmBgIGSF3vf3uB+4x+NBb28vbE4XMsvqMGvxoCiDoCZHCrmkEAI+D9lSITuYQqVS4cyZMxHlxJEgESGmb5hvNpvR2dkZVpi/khDyeahQBl6pufm92+3GyZMn4XA4vObE0yjG3wIWDAG61x4A8BYh5BGGYR5Y+vp+AJ/DokimFsAOLE5z2RHOeVac6JGaNYYqxnk8HnaPOJCBRbhwuVzo7u6G2+0Oq0LPBZfodrudDfmtyMSHowZ4iAeXbVCgWJ6G3IxPVwUqZFlYWEBdXR2bE09OTnop3JRKZcxurYleYYVCISQSCerr60OG+bFGLsn0qReJRCgvL2elx3SqTFdXF77//e/D7Xbjb3/7G3bs2BHy/vPXvYbFQYp/t/TvZwC8g0Wi7wPwuyVJ7IcMw8gZ70EPAbGioXs4oXqg3/UHm82Gjo4O5OTkYOPGjTHdyDTsLy0thdlsjlheSYlO83Ea8mvG9ABDAAKEc2/65sQmkwlarRZdXV3weDzsFlEgf7OVBLcYF89q/krCdw+dG8VUV1fjF7/4Bb797W/j97//PV544QU88cQT0Zwmn0PeGQB077YYwCTn5+iAxeQSPVzw+XzMzc1BrVaHpVU32VzQmh0oU6SBz+fD4XAs+xmtVou+vr6wc+hgoM0tNOyfmJiI+BgMw2BmZgazs7Mo27AZ4vTFLbjmkixIhDyI+AwKM/0/7QMV8riNLBUVFXC5XF7+ZtTUIdyiUaIRrOoeSzU/kvPEG6HEMhKJBJWVlfjVr34Vl/MRQgjDMDGHK0knOi1sEULCCtUNFie+fbgTBqsTV23Ox54aideKTgjByMgItFptRDm0P1Bt/vz8fEQ6el/QIYFOpxNpxZvwWp8eIsEC9jcXQi4VorE4i236iQUCgQC5ubnIzc31Mm0cHByE3W5nRzD7C42TQY5IzhGomt/f3w+73e7Vguv7XlZDLzpFnMQyszQkZximEIB66fWoBiwCSQ7daahOVUvhEGlSb4XB6oJUxMdHo3rs31jC5ui09VUmk2Hr1q1hfdiBbj56rIyMDL/NLeGC5uNCoRC1tbX4UGWDRMiH1enGgs0J+VKDRKxtqv5+x3foIm1bHR0dZbvbaNtqMhCL1p8b5nPfi78wP5lET9LwhlcB3ArgkaX/v8J5/W6GYZ7FYhFuPpz8HEjiis6tqhuNxrC3yDbkpaOxOBN9M0bcdmEJu702Pz+Prq4u1NTULJOfBgIt5Pl+UAsLC+jq6vIrZY0E9Jo2btyI2dnFqbdbyuR4d1CL0uw0FGSGX7GPFVwyAMvbVoHF1SkzMzNhI5bjFTX4vhffMF8ikcDhcMBqtUbtzhsuwvF0j2RFD9C99giA5xmGuQPAOIADSz9+HItba0NY3F67LdzzJJzo/qrqFosFdrvd788Pqk34c8cMtpbLsas2B2IhHw9ds4n9vl6vZ6ucra2tEeWiVGzD/aCoL1xzc3NMIRc9Dr0mtVoNj8eDnCwxrmuN3G8u3lVk37ZVqu7jFvWUSqVfJVi0SFR64Bvm0xnx1KAxWJgfKTRGO3QWJ6pypBDyeXE3nQjQvQYAl/m+sFRt/3rYB+cgoUQPVFUPVjn/52N9WLC68O7gHGrzZCjM+nQVdLlcGBkZgd1ux8UXXxxxgwV3a44rZd22bVvUNwR9kNH+eHqc1dzUwjCLs9fkcjmUSiVcrkXLaaoEk0gkbJgfywqZLGdWiUQCmUyGzZs3s/URbphPH2KRVvPnTA78+19GYXO6cUFFNg5tLUqKu0wikLAcPZgAJhjR04R86MxOSIQ8CDgeayaTie1v5/P5UXVR0bCfSlnp9NNoVx2Hw4GzZ89CqVQu2x5c7U0tXNAptXl5eSCEwGKxQKfTLVshI+1eS+aUFnoeSmw65Ye2q6pUKiwsLLDtueE8xAxWJ+xON6QiPlQGK4DwinFUUbeaEPcVPRwBDJfoDpcHPAYQ8Bf/eP+8pw7vDc2hvjCTFZJMT09jdHQUDQ0NEIlEMBgMUV0bj8djjQfr6urYnC8c+N60NB8PNJppNa/ooc5Np5GWlpay9RCtVovR0VHWi06pVCI9PT1kg0eyiB7oAeRrR+X7EOO24PpGdZVKKS6pUWJcZ8H+5gL2XMGiv9XYogrEmehOpxMff/xxSAEMJfonY3r88/F+pIv4eOyGBhTL01CQJcGNW0oAfPrQsNls2LZtG4RCIRwOR1Tda/RDHh8fj3gbztdE4ty5cxgfH0dLS0vADzUUWbVaLSYmJtjwOdFFpGjhu99ts9nYsJg6qdAw31cFthIrejD4PsS4Yf74+Dh4PB5GrRJ8ovbgoppcXNOQh+tavNuZqad7IKyL0F0oFKKhoSHkG6VE/3PnzGIxxeLEyXEDiuWf3uxWqxUdHR3Iy8sLO78PBJfLxc79rq+vj3ivnZvbDwwMsNFKsCd7IKJTG+jZ2VlUVVVhYWEBAwMDcDgcbIgslUoTvqJHS0KJRIKioiIUFRWBEOI1ghgAq9TLzMxc8ZHJoeAb5lttdvzqhW6IGBeePzGIbPsMKgtzvML8RBtDJgpxz9HDeZOUrJdvysWpCQNkYj6aizPZ78/NzaG/v9/vRNVIjSeolLWsrCxq91Mejwe73Y6enh4oFIqw5LX++tE9Hg96enoAAFu3bmUbImiIbDAYoNVqMTQ0BIfDgampqVWjcvMHhmGQlZWFrKwsVFZWspNIqX0TnSFns9kS2sQSr6KfRCxCfUk2hjRm1Cjk2FxbCNOCwSvMt9lsQZuk1kXoDoSXXwoEArhcLlxSm4PG4iyIBDxIRXwvk0XqGuvv+OGCSkMbGxuRmZkJk8kUlTuN2+1Ge3s7Nm7cGPaoZN+/g91ux5kzZ1BQUOB3sivXrpmKdwCwKjca4idiBlu8wJ1ESgiBSqWCRqNhm1gSNUcuFqK7PAR9MyZkSgQoU6Thq5eUY0JnRVGWGOliAbKzMrzC/IGBAQwPD2NiYsKv+WS0oTvDMBvhPZGlCsAPAMgB/D0AzdLr3yOEHI/0+CumdaeEo0oxh8OBjo4OZGVleZksRoNAUtZobKimp6dhMpnQ1tYWUfGO249OC3dcHX6ohyGfz2f76LkFsZGREQiFQvahsJpXe7FYDLlcjsrKymUDJujccYVCEbKoFwqxEP3PHTN4vWcOAj6D+y6vQqVSitq85SsyDfNlMhkqKiogFAq9hkukpaVheHg4amNIQkg/gBYAYBiGj0Vp60tYFMX8jBDyWFRvcAkrQnTf1Y52eAWqYEcC7gPDV8oaCdFpu6vVakV2dnbEoSd9j3THIBJxj+/fx7cgRjXtQ0NDbCipVCoj2v5Kttbdd8AELerR0cqZmZnse4i0tTgWok8v2CHgMXC6PdCaHagM0pcOfJqj+6vmv/fee+jv78fnPvc5XHLJJXjssceiLbJehsWpqePx+oxWJHSnIIRgYmIC09PTEavc/GFhYQGdnZ2ora31u5cZLtHpw0Iul6OlpQUdHR1RhfwzMzNgGCZujrgUXCMEX027QCDwWu1Xst0z2MOEW9TzeDwwGo3QarWYnFzswqQPtszMzJDvIRaiX9tcALvrHHLSRWgqygz58/6KcbSa/7WvfQ3PPfcc/va3v+H06dOx1CUOAfgT5+u7GYa5BYujl75NCNFHesAV8ykihLDNH9u3b4+5mEIlqMG2vMIhutFoREdHh9fDItJ9bZfLBZVKBbFYjLa2tojJFsn5fHXgNpuNDfGtVmvQfeJEI9yogcfjsUU9YHGbVqfT4dy5c+jr62NbVpVKpd+6Tai97WAozJLgm7urwv75UIIZj8eDtLQ07Ny5M6rrYRhGBGAvFqe1AIsuMv8CgCz9/9+xOGE1IqwI0Y1GI8xmMyorK6OaO851kaUWTS6XK+SWVyii0zC7ubnZq6ASSchPZ69lZWWFtRr5QyyrsEQi8VrtaW4/NjbGrvYKhSIpgpxo0wOhUOgVFtOWVe7UWKVSiaysLLbek8zutUDnitPf9HMAThNCZpeOOUu/wTDM/wVwNJqDJiR0Dwbqhy6VSlFYGNw/OxAo8agENT8/H+Xl5SHPHci0ghCCgYEBmM1mVpjje75wPkRqftHQ0BBRh54/xOOm8d0nttvtrMLNYDDAZrPB5XIlbLWPRx3AnzONXq/H3NwchoaG2EKrQCBISt0hlMNrHBxgbwInbPexiroWQFc0B03aik5XXqfTie3bt+PUqVNRG91Th5rh4eGIpKz+Vman08muwK2trX4/pECOrhS01jAzM8NuC5rNZrhcrsjeGOd8iYBYLGbz4v7+fqSnp2NhYYFVhdHcPtYqOEUiVlo+n4+cnBx2Vp7VakVfXx9mZ2dx7tw5ZGZmsqnMSqQqsTygGYZJB3AFgK9wXn6UYZgWLIbuYz7fCxtJ+UvQLrbCwkKUlZWBYRgIBIKwie72EHw4qgMDYHtFNmw2G0ZHRyOWsvpaRQcbqeT7e8FGLlPfeK6BZKx69USH1nSlLClZlBvTfnUqbaVV8FgIk4wVNi0tDTKZDDk5OcjKysLCwgIrL2YYhs3t4+VDF+xzcTgcMY2kJoSYASh9Xrs56gNykPDQnYpWYpm48kavGk+/Pw4AuLyEYJPMg6amppikrDMzMxgZGQk4UomLQKTlurv6pg5roamFe73cfnVuFXxiYoIt+EU6bTWZWncejwcejwe5XM7eZ/586OjDKxEz4lerKg5I4IpOCMHg4CAWFhYCTlwJexCDzQW3xwOLxQxBWgEUivSoiEBX9IGBARiNRr/5eKDf813R6VZeoJHLochKHWIzMjJWZEhDsGvjVsGrqqrgcDhY0ptMpqCNLL7nSCbRfeHPQZdb1KPKNrlcHlaKESoVidRdJplIyB1mt9vR0dGB7OzsgP5rfD4/7Bx2Sz6DjgwbijeW4tDF1Rgb6o9qX9vj8UCtVqO4uDiibS9f0tJoIFT3WqBrdLlc6OjoACEENpttmdJttVkdi0QiL3caGh7TRpZA4fFKE50LroNueXk5XC4XDAYDNBoNhoaG2LFLtJOQYRZFNBqTAznpIogE8XeXSSbiTvSFhQVWF+5vpaMItKITQvDbExM4NWHAoa3FyPPoYDQa8U8HLmZXj2g62EwmE3p7eyGVSlFbWxvR79IVnUpraZQSbDULtKJbrVacOXMGZWVlbIhJ942p0k0ul8PlcoW8sVYC3EYWAMsGTGRkZLCEWU1E94VAIFhW1KMNRTabDZmZmXhjkmDaDJQppPj7nWVhGUOumxVdIpGgra0tpPQvEFnHtBa81j0LiYDBvx/vxKOfK1m2+kaqWZ+dncXw8DA2bNiAmZmZ8N8M53xOpxNnzpyBVCoNKxrwR3Qq9a2rq2PdSz0eD9sMQguC8/PzmJ2dxalTp1hbJ6VSGfcOsHiQ0Dc8prl9R0cH+9ASi8UJHcoQj+p+Wloa21vg8Xig0ekxdGoEaXDgrF6PvgI38rODv4d1taKLxeKw/uiBiK5MF0HCBzS6BbSWK/yuvuGu6L4rsNPpxLlz58J7Ixw4nU6oVCrU1taGLfDxJfr09DTGxsbQ3NwMsVjM7jwAYKMFSvy0tDSIRCK0tbXBZrNBr9ezHWDU/yzWySyJKPYxzKfjiCsrKzEwMAAej8cWw2QyGZvbx9N5Nt7beDweD/k5Sly/HXh3SIddxTIoM8DaUXV3d/sdixVLjs4wzBgAIwA3ABchZCsTYNhiNMdPumCGgpKVEIIpgw3KdBGkIj4WtLP4YrUL8uIGtJT7D/3DmajqdDrR0dGBjIwMdgV2u90R5/a0altQUBCRio+KbLgPm7a2NvB4vGWiCnqT8vl8mEwm9PT0oLq6GsDig5O7YhoMBnYnQyqVJrSKHCu48lx/46QoWaJVEFLEi+jzVife6NUgQyLA5ZtycWmtEpfWfrrblZaWhtnZWRQVFS0biyWRSDA/Px/riv4ZQsgc5+tAwxYjxopp3fl8PpxOJ3713hje6FEjWyrElxsEkPAJLt+5I2guFGpGuslkQkdHB6qqqlBQ8KkVUKQhP224qaysjGqggtvtxtmzZyGRSNDc3MzmrIFuaq1Wy46CojcMd7V3u93ejihWK/R6Pbq7u+NKnHiBm6Nzi2EVFRWsSQVXz05TlEhX+3CtpELh9R4NTo4Z4AZQkClBS4l3kwvVffgbi/X666/j0UcfRUZGBrKysnDdddd53XtRItCwxYixokS32Wz4YMSANCGDSY0BJhRje9OGsKSsgXzh1Wo1hoaG0NjYuKwvOFyiUxWf2+3G1q1boVarYbVaw39zWCxSqdVq1NbWstXqYCSfnJzEzMwM2travG507movFApZwhNCkJaWBolEwh6fup329fWxIpJAW2ArPZLJ16TC3/DISHzm47GiZ0oEcIOAzzBIEy4/nr/IgY7Fuvnmm2EwGNhFiEaBEYAAeINZnLP2K0LIfyPwsMWIsaJEd7lcuGZTFp56fxytZUpc0lQbdreTL2GpO43BYMDWrVv9rgrhEJ3q53NyclBRUQGGYcLWulPMz8+jr68PmZmZbMgdrBGCesa1tbWFrLJTYQjw6WpPUxJuvzd1O6VbYPR7kQheYkW4DxPf1Z76zFNLqlDda7HUG+wuDxwuz1K4noOCTDGkIj42+DGfCMcvrrm5GTfeeGM0l7KTEKJiGCYPwJsMw/Rxv0lIbMMWVzRH1+v1qODx8PLXLoqoQd+3GEf3pdPT04POTQtFWNqi6muAEUrrzgXdY6+vr0d3dzcmJyeRm5vrt9fe5XKhs7MTmZmZ2LAhdCTj7/0AYG8+utp7PB6vOWw0vKSCl8zMTNhstpiabsJBtKaNvj7zZrMZWq0WPT09bPoSj6kyWrMD//HOGIw2F27aWoRt5XK0lgb2gwtnHFO0OTohRLX0fzXDMC8B2I7AwxYjRkJW9FCqMJfLheHhYTidTlx88cURf1jcYpzZbMbZs2dRWVkZdTcc8OkWnG+LKj1fKKKTpamuer2eXZnb2tqg1WrZSS4KhQI5OTmQy+WsuUVpaWlM1+17nf5We+7UVYZhYDQaMTAwgP7+fohEooSJdeKRO3O716jQRa/Xs1Nl0tLS4HQ6YbfbIy5IjuusMFicSBfzcXLcgG3l8qA/73a7g2ononWAXWpm4RFCjEv//iyAf0bgYYsRI+mhOy2UFRYWYn5+PqonMs2DfM0fowE35A8kggn14HK73ejq6oJIJEJLSwsbstJVlbq86nQ6zM7OoqenB06nE6WlpTHPcg+EYKu9TCaDRCJBeXk5xGLxMqMK+jCKRaxzYlSP/3jPgMs3S3DHzvjtLfuOirZYLGhvb/eStYa7/VidI0Vehgh6ixO7akJ3QLrd7qBahhgEM/kAXlp6KAoA/JEQ8jrDMCfhf9hixEgq0WlY29jYCKFQCJ1OF9VxeDweDAYDLBZLTHPMqd+7RCIJGfIHWtGpu2thYSHrde6v6Mbn85Gbm8uaQWzatAlGoxFnz54FAFallagcmq72tAdBIBBAJpOBEMKaPABgfdqHh4chFovZ1T5S77PvvNQLi92FsRPT+MymAlTlxN/Eklo4iUQitLa2srJWugCEEhtlpQnx3Str4PYQCPnhad1DKeOiNIYcAdDs53Ut/AxbjAZJCd09Hg8GBwdhMpnYVdPpdEaVI7pcLvT398PtdmPLli1R52hcKWpxcXHQnw2U23MbW7KyskIW3cbGxqDX67FlyxYIhULk5OSgsrKSbRoZHR2F2WxGVlYWcnNzoVAo4m6L3N3dDYlEgsbGRrb2wBXrcAUvdrsder3ea8AEtZwO9XeXpwlhdbjA4wEyceJkvNzPhStrpau9Tqfzspv2vX4ew4DHD+/BmsgcPdFI+IpOq9jZ2dle0tFo9Op0GENRURHm5uaiJrler0dPT4/fAZD+4K8YR3P6xsZGtgki0EpMt+t4PB5aWlqWXTe3aYSu+NRYQyQSITc3Fzk5OTGNbaICory8PJSWlrKvBwvxfZ1O5+fn2SaQtLQ0drX0lxv/+otNeOYvHbiitQp5GYkT8wR6uNLVnjtDjutMI5FIvJpYwkE4VfdoVvRkIKFEDzaIMNItK41Gg4GBATQ2NkIikUCtjq4A6XA40N/fH5FpBTd0J4RgdHQUWq2WLboFIzkdxpCbm4vS0tKQYTnX/qm2thZWqxUajYZ156EFvUgksHR6bGVlZchJn8G277i93lSa29PTA4fTiSOjPPRo3fjG7ipcvikX+ZliXFkpwqaCxK5w4arifJ1pLBYLtFrtsnFYwWoToYwhrVbrqp2hlzCiT05OYmpqKmYbZy6xNjW2QpouAQMScTRA56HTkD8S73Caing8HnR1dUEgEHgV3QKR12w2sw420frVp6WloaysjPVL02q1mJ6e9hLF5OTkBHw/RqMRXV1dqKurCyt64SLQak8IYYU6BQUFOKtawF/f6YPd5cGDr/ai0K2GUqmE2+1O+J59tJV930IpHYfFrU34jsMKlaNHu52YDCSE6ENDQzCZTNi+fXtMOSYtlonFYkwKivDky30oU6Th/s/WRiRl5c4xj8YPjcfjweVy4eTJkygoKEBxcTH7oQY6lk6nQ39/PxoaGuIWzvH5/GVqMo1GgzNnzgBYLOjl5uay75GOB25qaopL+2Sg1b4gUwIej4FYyEOVcnHvXq/XszssNMRPRAdbKPKFA9/hErRl1XcclsvlCniuZLXkRouEEL28vDwoCcIBtU2mxbJfPt8BhVSICZ0V5+b9y1/9gfrC1dTUIC8vD1qtNuLGFrPZDIPBgJaWFsjl8pAruUqlwrlz59DW1pawZhOumoy6wMzNzWFkZARmsxkikQg2my0ugzH8gbvaV+SJ8IfbWtFzbgEXV8khFfEhlUoxNzfH7i7QfvV4eNFxES+dOxfcllXuOCzauRZoHFY0ZJ+cnERZWdnbWNxiIwD+mxDyBMMwDyEOM9coEkJ0kUgUdmjt749D8/GGhgbW4ODyTbl46cw0KpVSFMslmPV3MB9Q3TvXFy7Sxha6VZOens5eS7DK+tDQECwWS1hy1nhCJBKhqKgIhYWFGB0dhUajgVKpREdHB8RiMbvax6OnfUpvxd3PdcJDgP882IByhRS1eRmozVuMXKjWm44uoukFFetQlV48nGcT7enOHYdlMBiwYcMG6PX6ZeOwov27Lj3svk0IOc0wTAaAUwzDvLn07Z+RGGeuseeJx0F8EWmrKn2y03xco9FAVroJw/MEzRkEfB6DPU2FuGxTHiQCHni84MfnqtR8RTDhEp1uh83NzaG1tRWnT59GV1cXWwH3zYmpaCY9PR1NTU0rEsYRQtDX17fMkdZisWBubo4V6igUCuTm5iIrKyuq63zqgwkMqs0AgP9+bxwP76vzuoaxsTEYjUYv+2yuWEcmk6G0tJTd96bz12iIHOkMuWTmxTS3547Dmp2dxa233or5+Xn87Gc/w9VXX40NGzaEdbylhqTTALCkjOsFEHy/NwqsWFML4E10mo+LRCKICjfgsbeG4fEAB7YWY1/zokRUKgr94bvdbnR2drLjkHxvgnCITvebGYZBS0sLAGDHjh0wm83QaDRob28Hj8djV0k+n4+Ojg6UlJRENXkmHqDvm+6BcwkslUrZgh5tGFGpVOjt7UVGRgZycnKgVCrDLlBuLsyAWMgDA6Ch6NP6A23QcblcaGxs9Prbc3N7WtTztXMyGo3sDLlwJ8Ymc0qLL7j99seOHcMtt9wCmUyG48ePh010LhiGqQDQCuAjABcjDjPXKFYF0Wk+XlpaipKSEvylTwOnm0DAYzC7EH4+brVacfbsWTa/8odQRHc4HDhz5gzy8vJYIQ3Nx7k5sd1ux9zcHHp7ezE/P8+GxStx49FiY3FxccgHjW/DiNFohEajYUNpSrxgofQNbUUoV6TBQ4DtFXIAn/rbi8Vi1NfXB1UZ8ng8CASCZb323A42h8PhFSJTaauvWGclic6FyWSCUqnE3//930f1+wzDyAAcAfANQsgCwzBxmblGseKhO7UR5ubjF1YpMKQxw2hz4doW/w0fVMRCP2S6p1tfX88aMwQ6ZyCi0ypxTU0N64oSqOgmFoshEonYmW92ux2zs7Po7++HTCZDbm5uRKtktKDDMWpqaoKacfoD1/qpurqaHdlE53xnZ2cjJyfHbyi9reLTvzGNJuRyOSoqKsI+f7Bee9qvThtxqIiIil3oap8soofSfMRiDMkwjBCLJP8DIeTFpfPFZeYaxYpOUzWbzTCZTODl1+K3n2hweR2DzUWZSBPx8eVLKoL+Pl2ZqSfZ1NRUWKaUgVpONRoNO2iChorBim4TExOYm5vz2pOnjRbcVZIKNQK1qsaC+fl5VuEXbVMPF9yRTTT/pEo4iUTCrvbcwhNV3NFtx1gQTKxDnWcZhmHFOr29vbDZbBCJRDAYDDG3rQZDqOp+tMaQSw+QpwD0EkIep68zcZq5RrEiRKeFK4/Hg6Lyajz4xgR4YHBqYh5P3dIaVoMBtaKie53btm0Lq4DjG7oTQjA+Pg61Wo3W1lYIBIKQclZa8GptbV12Y/mukjabDXNzc16tqrQQFstNqdFoWG/5RKixuPknsHgjz83Nobu7Gy6Xi+0QGxkZCUtxF835Af9iHeqjl5+fD41GA4PBgJmZGdakIlpLqmAIp6ElGqK///77AHAzgE6GYc4svfw9ADcxcZi5RpH00J02k5SWlkIqlYLHAGIBDyabC9npIvAiGKrQ0dGBnJwcbNq0KaJ0gRKdOzettbWVPW6gYzmdTnR2dkKhUIQ1vRVYtL/m7slS55S+vr6oCmEAMDU1xdpOJTo1oKC6cdoTPj09ja6uLgiFQqjVahBCQk5uiQX+Vntatc/KykJeXh4YhmF99LgGlDk5OTGLdcIZ3hBN6L5z504QQvxdWNR75v6Q1BWdjhWmzSSjo6MQMh788OpN6JleQHOJHPwQW2fAYi6t1+tRW1uL8vLyiK7Bd+RyTk4OW7gLRnKLxYLOzs6YVi/aqkpD/IWFBczNzbEhPv1eoBWa9s6bzWa0trau2HAHm83GpkqZmZns+6BTWbmpSqJabunfgj5IuVbZaWlpbPqh1+tZu+lwR0n5w1ruXAOSRHQaHs/OzrJjhYFPq+7lSinKleHlr1QEQyuwkYLH48FisWBkZATV1dVeE0UC3ZQGgwG9vb1xy4UB74knNMTXaDTo6+uDw+FYttdNow+hULhi+/TAp38LrqyW+z7obsTQ0BCsVivkcjlyc3ORnZ0dt/yZ/i0kEgmqq6vBMIxXiE//Az4dF8UwDLtATE5OsqlJqB0GilANLat5eAOQBKK73W50d3eDz+d7iTiAT/PscMBtbtm6dSuGhoaimr9msVigVqvZwl0owcX09DQmJyfR2toa90kpXEgkEpSWlrJNFlqtlt3rlslkMJlMKCgoQGVlZcKuIRQogYP9LcRiMYqLi1lBCW0NpUYQVHAUrTSYNhZlZGT4/Vv4C/Ep8dPT01kfPdq2SsdEU4Vbdna2X2luqBzdYrGw9YzViITm6HRfu7i42KsHmoJaPocCLd4JhULWbCKafvbx8XHMzc2hpKQkZA85DQ1NJhPa2tqSOvGU27xis9lw+vRppKenQ61WQ6/Xs6FxMlsiZ2ZmMDExscyOOhD6Z00YVJvxdxuU2LgUNVGFXldXF9xuN5RKJXJycsL2ofd4POzwznBSNn8FPVrU47atMgyDhYUFlvgCgWCZj16icvRkIWF3r06nQ29vb9B97XDIarPZcObMmWUPi0g067RS7nK5UFtbi5GREdZ7zN/KRKMQOnhhpcJkk8nEtphyhzZQoY7T6WTJEq2cNRxMTk6yUVA4D7wJnRVf/sNZuNwEf+6cwa++0OxlBFFeXs4OlpycnITRaERmZiZbmPR3DjoMg/b1R4NwxDrl5eWsWGd4eJidH0d9BwKBDphcrUgI0QkhUKlUIc0dQhGdDiX097AId0WnwxGVSiXKyspACIFMJmO3itxuN7tCymQy1p21sLAwoLouGdDr9WybKzf3S0tLY0N8l8vlFeJnZmayFlTxiEBoukR16+Hm2HMmOzyEgIBApfcfsQmFQi/3moWFBWg0GoyPj3uttunp6XC5XDh79mxc9uopIhHrLCwsYHJyEmazGUaj0a+PXrQOsFwwDHMVgCcA8AH8mhDySEwH5CBhoXtTU1NINVEwsqpUKjZU9BeihjN/jVpBV1VVsT5i1J2Var+dTifm5ubYG9rpdKKiomLFNOvAok3V+Pg4Wlpagj4oBQKBX6snqhWnVfxoagtc3Xqkxb+W0ixc31qI9skF/MPfha4pcAuTAFjtweDgIKxWK5xOJ2u1lSgEE+tkZmayElyFQsE+hOnQS6fTGdOARQBgGIYP4BcArgAwBeAkwzCvEkJ64vH+mBBkjHoyhNPpDBlam0wm1kudPSEh6O/vh81mQ0NDQ8CVaXJyEoQQlJWV+f0+3cpraGgIqXQDwN5YpaWlMBqNMBgMrJQ1JycnaTn6+Pg4tFotmpqaYjontaCam5uDy+Viq/jh5MNc3XpNTU3InyeE4N1BLawuDy7flANBnKrrTqcT7e3tUCgUrJ+7VCplV/tkDZZcWFhAV1cX6uvrvR6a9OH6y1/+Er///e/R3NyMAwcO4Atf+EKkD1eGYZgLATxECLly6YXvLp3jJ/F4Dyva1CIQCLxWZafTyRpJbty4MegNFmz+Gh2O2NraCqFQGLToBiw+NGZnZ7Flyxa20ESlrGq1GuPj4zGvkKFAV1Cn0+nXQDJScC2oaIhP82Hq3a5UKpcVmNxuN1vwCle3/lq3Gv/y2gBAgJE5C762K7zfCwbaXFRZWcnacFHZ9NzcHDo7O9kxVLm5uQmbv242m9Hd3c16Gvjz0bv//vvx/vvv43vf+x5OnjwZrb6hGMAk5+spADvi8iawSrrXgE8bSqqrq1mP8WDwV4zzeDzo7++Hw+Fge6FDyVlpeOrb0sqVstbU1LArpL+8PtYbjBb/pFJpVKOZQsE3xDcYDKwbjVgsZqMWPp+Ps2fPorCwMKJc+Ny8DS73Uk5uiGwYpT9Qr/yamhqvARcMZ2oLnchKm6JoQS+eNQrqOtzY2OhlXAJ4V/I/+OADjIyMoK6uDrt27Yr5vIlAwogezs1KBy1SR5mmpqawK5e++T2NBuRyOWpra0OKYOjcs6ysrJDRA+C9QnLzerPZjOzsbOTl5YXld+4Let0FBQVJKf4xDOPlMmuxWKDRaNDZ2Qmj0cjKRSOxRTqwpQj9syZYHG7cfWls+/x0l2Xjxo1BuxCBxYIed3Y87XCjNQoa4kfTTERJ7lsM9cXp06fxne98Bx9++GHUBqBLUAHgbieULL0WFyQsR3e5XCGLZYQQvPPOO5DJZGhubo6oCUGr1UKtVqOurg4Wi8UrzAtFcqvVio6ODpSXl8c8w5qKQtRqdcR5Pb2OqqqqWG+SmED1DlVVVfB4PNBoNDCZTAEHSdicbrzZp0FZdhqaSwIPJYwUtOV206ZNUakeuaAFPY1Gs2zuXaiHMfVHCGXseebMGdx111148cUXUV1dHcvlMgzDCAAMYHEyiwrASQBfIIR0x3Jg9gQrRXQarmo0GnzmM5+JeCU0GAxQqVQoLCxEb29v2EU32tpZX1/PVnnjBW6L6tzcXNC8nhoNJuI6IoHJZEJnZ+cyeS8dJKHRaKDT6dgQPzc3F9/98yD+NqwDAwZP3dyMhqLYZcF0BY2nzJiCquBop1t6ejq72vsuLvRhs3nz5qAk7+rqwp133onDhw9H5SbjAwYAGIb5PICfY3F77TeEkIdjPTB7gpUgOh0oUFhYiKmpKVx00UURH5/6lTMMg8bGRohEopBFt5mZGYyPj6OpqSkpqjKa12s0GlYJlpeXB7vdzppWJsKhNVwYDAb09fWhsbEx5NYQLYLNzc3h+++aMG3xQMzn45/3bMRVm0PXVIKBPmy4uXCiQAt69GEMgBUdCQQCdHR0oL6+PujDpre3F7fddhueffZZ1NfXx+OyEq7IShjR3W43XC7Xstfp9Ja6ujooFAp88MEHEROdEIKuri7Mzc3h4osvDll0o8KP+fl5NDY2JlXOSkHzejqjvLCwEPn5+XFt9ogEVLceaq/eH06P6/Cvx/tRkEZwUy2Qo8hmG1cirTjTyCZe3vORgn4us7Oz0Gq1UCqVKCoq8rsjAQADAwO45ZZb8Ic//AGNjY3xuoyEEz2pd/y5c+cwPj6+zGs8ksIPVUlJpVLWo42aRfgD3RMWCARobm5eMX8xgUAAq9XKmlYuLCxArVZjYGAg6fv1MzMzmJycDFu37ou2cgVevOtCAPDrRBNu48r8/Dx6e3vR3Ny8YpGNUChEdnY2xsfHsWXLFhBClhX0aF/B6OgobrnlFjzzzDPxJHlSkJQVne4RWyyWZSvqhx9+GLY7DC2SlJeXIycnB8PDw9BqtUhPT/drw0zlrHl5eQGFNckA1dozDLPMJCOSvD4eoLr15ubmuD9UaOMKfS+EEHaf23cbkqrLmpubV3ReGa3y+ysA0r6C4eFh3HvvvXC5XHjggQdw++23x9tgY+2G7h6PB06nk/UUo/vRvivvyZMnw6q4cyeg0rne1IDAZDJBrVZ7EUUmk6G/vz8q08R4ggpQqHFiqMglUF4f6349V7fua8WcKNBx0BqNht2GpFLkkZERNDc3J7T1NxSCkZwLlUqFgwcP4tChQxgdHUVRURH+6Z/+KZ6XsraJbjAY2G2bQNtY7e3t2LhxY9DQTaVSYXJyMqyim9VqxdjYGKanpyGVSlFQUMDOI0s27HY76/cejU6b5o9cokRj4kAjKrfbjbq6uhXpxqPbkBMTE9DpdFAoFMjPz/db+U4G7HY7e+8F26+fmZnBDTfcgJ/97Ge49NJLE3U5azdHNxqNOHPmDBobG4NWMIM1tnBDfjpbPVRlXavVwmQysUU6Ksax2+1svhVu/3MsoJNUa2trvdRdkUAoFHrNTaf79ZHk9VzdeiJUd+GCNiE5nU5ccsklcDgc0Gg0OHv2LIDlAyITiXBJrlarceONN+Lf/u3fEknypCChKzod9hcM3d3dKC4uXhY6uVwudHR0QCaTsUKOYIMbCSEYHByEzWbD5s2bl+X8VO+tVqthMpliUrOFArVbiuckVS7Czeuj0a0nCtPT05iamkJLS8uy/JYOiNRoNKyffLztp7jnam9vR21tbVBHmLm5OVx//fX4l3/5F1x11VVxvQY/WLuhOyEEDocj5M/19fWxww4oqFNsWVkZq88OtpLTAQIymYz1EAsGXzVbRkYG8vLyAm6pRAK1Wo3R0dGk5p/+8vrs7GwMDw+jqKgobj3c0eLcuXOYnp4OqwBIPxuNRgO9Xh9U3BIpKMl9NfS+0Ov1uO666/Dggw9iz549MZ0zTJz/RB8cHGTtegFvswmu5joQeW02W0xzz6jpgVqthlarhUQiQV5eXlQ3Fq1oNzU1Jc2G2RdOpxMzMzMYHh72cpZN5n6920Pw9sCiGKUmzQqddg7Nzc0RP0RpoZUKdQCw6UqkIX64JJ+fn8f111+P++67D9ddd11E1xsD1i7RAQRsI+ViZGQEaWlpKCwsZPfZGxsbIRaLQ+bjVGyxadOmkA0Q4cJsNkOtVkOj0YRlwQx8Oi6Zpg0rOQuM6tY3bNgAuVzutToma1TUc6dUePLdMbhcblxbK8I3926Py9+E5vVzc3OwWq1siB8q/aItr9SAJBCMRiNuuOEG3H333Th48GDM1xsBzn+iUy9wm80Go9HIEiUUydVqNUZGRhIqI6UWzGq1mm1NzcvL81pN6ORViUQSlklDIkFlwf704sncr//FX0fxh48mQDwEN19Yjq/F2NHmD/706/60FNS8IhTJzWYzDhw4gNtvvx0333xz3K83BNY20R0OR0g7qYmJCUxOTiInJwfV1dUhQ3XqEa/T6dDY2Ji0EJludanValitVnZc0djY2IoLcoDIdOtAYvfrz/QM4OnTOiiys/GNy6ohT0vsZ0RDfPoQo0MksrOz0d/f72Ve4Q9WqxUHDx7ETTfdhDvuuCOh1xoA5zfRbTYbPv74Y2RkZKCxsTEkyT0eD3p7e1mF2UqFyG63GzMzMxgcHASfz2dJolAoVky3Ti25olmdY92vt7vceGdAi4JMMdIss3C5XCu2Xw+AnWw7PDzMGj0GalG12Wz4whe+gP379+MrX/nKSl3z+Ut02txSWFiIhYUFbNy4kbV98geqsMvJyUFZWdmKhsjUhnnTpk3IzMzE/Pw81Go1dDod0tPT2RsrGbp1um0VaT9/IPhWvcPJ6x862oe3+ufgcbvx3Z3ZuPrCxhX9fFwuF9rb21FeXg6lUgmdTgeNRoP5+Xn2/VDS33zzzbjiiivwD//wDyt5zWtXMAMsupn4I/r09DTGxsbQ0tICPp/PVkRFIhHy8/ORm5vrdVPRHuGqqqq4T+2MFDqdDgMDA14hMnVs4cpxqc8ctQ1OhJHh5OQkNBoNOwU2HuDxeKydMTevD+abN6G3wuF0gc9jIMkuWBUkLysrY+8V7rw7+n6++c1vsvWMz33ucyt6zclAQld0XydYOv1kfn4eDQ0Ny4puvhXvvLw8iMViDA8PJ8SQIFLQ8UzNzc1hEZc2eGg0GhBCkJubi7y8vJiLh1QrbjKZkqZbB/zn9Tk5OXi3YxjP9tlRV6zEd66sgSiMsdeJgMvlYif1BvMddLlcuPPOO1FZWYmqqiq8/vrr+OMf/xhVc43b7cbWrVtRXFyMo0ePYnR0FIcOHYJWq8WWLVvw+9//HiKRCHa7HbfccgtOnToFpVKJ5557jitiWtuhO5foVNRCq9Ph2D0NDQ1Bo9EgPT0dBQUFyMvLW5FOJ1oA1Ov1Ufez060htVoNh8PB5vWRupdSO2yPx7NiebDJ7sInYzrkCWyYU42BEMJGYivVX09JXlJSEtQezO1246677kJ1dTUeeuihmP9+jz/+OD755BMsLCzg6NGjOHDgAK677jocOnQIX/3qV9Hc3Iy77roL/+f//B90dHTgl7/8JZ599lm89NJLeO655+hh1jbRqcsMd6wSHWcbztwzs9mMhoYG1kBSrVbD5XKxK2MyGlW4xIpXATBaOe5q2MrzEIJDvz6Fc/M2iBg3fnltOWoqylZkv57C7XbjzJkzKCoqCto85Ha7cc899yA/Px8/+clPYv77TU1N4dZbb8X3v/99PP744/jzn/+M3NxczMzMQCAQ4MSJE3jooYfwP//zP7jyyivx0EMP4cILL4TL5UJBQQE0Gg29hrWdowOfFt02bdqErKyskCSnXnJpaWnshBA+n4+SkhKUlJTA6XRCo9Gwuna6t50IX2863JHq7eN1fK79Mi1+zc7Oor+/P6Acd7Xo1p1uD8b1VjBuF1w8HmTKgoB5fbhz32NBuCT3eDz49re/jezsbPz4xz+Oy2f5jW98A48++iiMRiOAxYYquVzORnwlJSVQqRaNXFUqFTszTiAQICsrC1qtNmkt1Akl+uzsLAYGBtDS0gKRSBRyRDFt6wymzxYKhSgqKkJRURG7MtLRtzQcjsfAQYfDgbNnzyZcK+5LEmrISBWDtELc09PDvu+VBB8E+6r4eHuKwdVNhSjM9K5VcP3wq6ur2by+p6eHzevjNXCBDl4MNa7J4/HggQcegEgkwmOPPRaXqOzo0aPIy8vDli1b8M4778R8vEQjoUSXSCRoa2tjJ1EG+2CNRiO6u7uxYcOGsOdMc1dGt9sNnU7HDhyUy+XIz8+PqjuNVvmTbVrBMAzkcjnkcjlqampgNpsxPT2Nvr4+pKWlweVywWazrZhZAx1YefvOKnwvTJtsXz987oM5li41SvL8/PygDz+Px4Mf/vCHcDgc+OUvfxm3+sH777+PV199FcePH4fNZsPCwgLuvfdeGAwGuFwuCAQCTE1NsYtEcXExJicnUVJSApfLhfn5+ajbl6NBQnP0Z555BlVVVew2WiBoNBoMDw+HreoKBd/utMzMTDYcDvVBUzvo1VDlpw+cDRs2QCqVeslxaTicjP5t4FO9eEVFRVy2OP11qdHPKFRezyV5sGiLEIJ//dd/xfT0NJ566qmYOxMD4Z133sFjjz2Go0eP4sYbb8T111/PFuOamprwta99Db/4xS/Q2dnJFuNefPFFPP/88/QQa7sY99JLL+GPf/wj+vv7sXv3buzbtw/btm1jyUYIYfeCqXtMvEHDYdqdlp6ejvz8fL9zuGnInCw76GAIplundQqNRgOr1cqGw4makU51DtXV1QmJcLj6A61WGzSv93g87Jz0YJNtCCF49NFHMTQ0hGeeeSah4iUu0UdGRnDo0CHodDq0trbi//2//wexWAybzYabb76ZHRr57LPPoqqqih5ibROdwmq14vXXX8fhw4dx9uxZXHrppbj66qtx9OhRHDp0aNncs0SBOzhxbm6ObUnNzc3F7OwsZmZm0NzcvGItphSR6Nbdbjfry7awsAC5XM5OV4nH35R68IcyaognqCmjRqOB0+n0cp+h6kha2PIHQgieeOIJtLe3449//OOKf55h4PwgOhd2ux0vv/wy7rvvPuTl5aG1tRXXXXcdLr744qR/IGazGbOzs5iamgIhBJWVlcjPz0/aOF5/iEW3zrVe1ul0kMlkbDgczYpGW17DmYOWKNC8nj6cMzIyUFVVFTCvJ4TgySefxN/+9jc8//zzK+JHFwXOP6IDwA9+8AM0Nzdjz549ePvtt3HkyBG8//772L59O/bv349LL700KR8QbZIRCAQoLS1lc2CGYdiVPpkhfDx169zoRavVQiQSsfqDcI5N6wN1dXUrOjIKWPycOjs7IZfLIZPJvPJ6bmsqIQRPPfUU3njjDRw5cmRFH9gR4vwkuj+4XC689957eOGFF/DXv/4Vra2t2L9/P3bv3p2QKjP1pFMqlSgvL/f6nt1uh1qt9ip8JVqgQ2sVTU1NCcknLRYLKy8GwD7I/MlxEzkHLVJwSc79nHzz+t/97new2+2YnJzEG2+8saI20lFg/RCdC7fbjQ8++ACHDx/GX/7yF9TX12P//v244oor4mIyYbfbcfbsWZSVlYWcpkoLX7Ozs3A4HKxAJx5z0YFPdetUBZiMWoXdbmeLeb7viXbmJWMOWih4PB50dXUhMzMzpEjoF7/4BZ577jnI5XKYTCa89dZbET+YbTYbdu3aBbvdDpfLhRtuuAE/+tGPotWvR4L1SXQuPB4PTp48iRdeeAFvvvkmampqsHfvXlx11VVROazSGzmavNPlcrHmExaLBQqFIiaBzmrQrXPfk9FohNPpxIYNG1BYWLiiHV2U5BkZGaisDO5Q88ILL+A3v/kNjh07xj6sonlI0QGMMpkMTqcTO3fuxBNPPIHHH388Gv16JEgRnQuPx4MzZ87g8OHDeO2111BaWoq9e/fi85//fFjztOkYoFDD7cMBFejMzs7CaDRGbB9NdetpaWlhOdcmGtSiury8HAaDAfPz86z+wHc+eqJBh2jKZLKQJH/55Zfx5JNP4ujRo3GtJVgsFuzcuRNPPvkkrr766mj065Fg7Wvd4wkej4e2tja0tbXh4YcfRldXFw4fPoy9e/ciJycH+/fvx9VXX+1XcTQ7O4vx8fGopof6A3ev11evnpmZifz8/IBbXFS3rlAoltUHVgL0Adja2gqJRIKioiIv/cHw8DDS0tJYQ41E7o4QQtDd3Y309PSQJD927Bj+67/+C8eOHYsbyd1uN7Zs2YKhoSF8/etfR3V19arVr0eCNUV0Luhc9MbGRjz00EPo7+/H4cOHccMNNyAzMxN79+7Fnj17kJubi48++ggikQhtbW0JKXT506vPzs5icHCQ3eLKyckBn8+H0+lkNfQrrVsHFhsxhoaG0Nra6lWl5spxaUirVqvR3t7OegXE21iSEIKenh6kpaVxxSR+8cYbb+Cxxx7D8ePH47r1x+fzcebMGRgMBlx77bXo6+uL27FXEmuW6FxQD7kHH3wQ3//+9zE8PIwjR47gpptuwvz8PIqKivDkk08mJfz0JQjd4hodHYVIJILFYkF1dXVUs9jiDY1Gg9HRUbS2tgbdcmMYBjKZjO3io40q3d3dcduVoCQXi8UhSf7222/j4YcfxrFjxxKmF5fL5fjMZz6DEydOrFr9eiRYUzl6JCCE4Itf/CIUCgUqKyvxyiuvwOPxYM+ePdi/fz9KSkqSmhdbLBa0t7dDLpfDbDZDIBAgLy8v7H3teGN2dhYTExN+RyRFAjpOSa1Ww2azsR2Ekcy3I4Sgt7cXQqEwZJ/9e++9h+9973s4duxYyB2TSKHRaCAUCiGXy2G1WvHZz34W999/P5555plo9OuRIFWMiwVdXV1oaGgAsHgzTU9P48iRI3jppZdgtVpx9dVXY9++fXHtNfcHf7p17r42Fejk5eUlZf93enoaKpUKLS0tcU1lqByXVvDlcjny8vKCdqcRQtDX1weBQBCS5CdOnMB9992Ho0ePJqR1uKOjA7feeivcbjc8Hg8OHDiAH/zgB9Hq1yNBiuiJglqtxksvvYQXX3wROp0On//857F///64Txyl1eympqaAoS13UITH44mbt5w/qFQqVtOfyEYPKsdVq9Ws64yvHJeSnM/no7a2Nujf/ZNPPsE999yDV199dcU99BOAFNGTAa1Wi1deeQVHjhzBzMwMrrzySlx77bWoq6uLScBCu+Ei0a37esvFU6AzOTmJubk5NDU1JX27jNtMJBaLkZubi/n5eQgEgpAP1zNnzuCuu+7CSy+9FO2KudqRInqyYTAY8Oc//xkvvvgiRkdHccUVV2D//v1obm6OiPTx0K37CnSiyX8pxsbGYDAY0NTUtKKz4YBFiW1PTw+sViukUikbwfjrK+jq6sKdd96Jw4cPY8OGDStwtUlBiugrCaPRiGPHjuHIkSPo7+/HZZddhn379mHr1q1ByTIxMcGunPEKj33zXyrQyc7ODkl6ag2dLIltMNA59h6PBxs3bvSKYLgtqTKZDP39/bjtttvw7LPPor6+fkWvO8FIEX21gNtT39HRgUsvvRT79u3DBRdcwIbBydKtcwU68/Pz7NhpX4EOddOlU15XWn1Hp866XC5s2rRp2fXQCObMmTO477774Ha78aMf/Qg333xzUlONFcDaJfrrr7+Oe++9F263G3feeSceeOCBaA+16mCz2fDmm2/i8OHDOHXqFC666CLs3bsXx44dw8GDB7Ft27akkYoQwha9dDod6yKrUCgwPDwMt9u9onPQuNc5NDQEp9MZ8npGR0fxxS9+Ebfffju6urqgUCjwyCOPRHzOyclJ3HLLLZidnQXDMPjyl7+Me++9FzqdDgcPHsTY2BgqKirw/PPPs5N27r33Xhw/fhxSqRS//e1v0dbWFsvbDhdrk+hutxsbNmzAm2++iZKSEmzbtg1/+tOfzsvwy+Fw4M0338Q3vvENSKVStLW14dprr8WuXbuSvj9OCMHCwgLUajVUKhUEAgGqq6uRm5ublDlwwa5reHgYdrsd9fX1QUk+MTGBgwcP4te//jW2bdsW03mnp6cxPT2NtrY2GI1GbNmyBS+//DJ++9vfQqFQ4IEHHsAjjzwCvV6Pn/70pzh+/Dj+8z//E8ePH8dHH32Ee++9Fx999FFM1xAmEk70hMSWH3/8MWpqalBVVQWRSIRDhw7hlVdeScSpVhwikYjVRZ86dQo333wzXnvtNezcuRNf+cpX8Nprr8FmsyXlWqjVssPhQFFREZqbm2GxWHDq1Cm0t7dDpVLB4XAk5Vq4GBkZCYvkKpUKN910E5588smYSQ4AhYWF7IqckZGBuro6qFQqvPLKK7j11lsBALfeeitefvllAMArr7yCW265BQzD4IILLoDBYMD09HTM17EakJDHPFfsDyw2AiTpybgiuOeee9gbePfu3di9ezfcbjfef/99HDlyBA899BA2b96M/fv34/LLL0/I/jjwaUecVCplRUAZGRmorq5mBTpnz54Fj8dLiFbdH0ZGRmC1WkPWCGZmZnDw4EE88cQTuOiii+J+HWNjY2hvb8eOHTswOzvLSpALCgowOzsLwP99q1KpVoVcOVacF1r3lYa/G5jP52PXrl3YtWsXPB4PPv74Yxw+fBg/+clPUFNTg/379+PKK6+Mm7kDdWLJysrya34glUpRUVGBiooK2Gw2qNVqdHd3J1SgMzo6yhYmg5FcrVbjxhtvxL/9279h165dcb0GYNGD4Prrr8fPf/7zZY45oeYNnC9ICNGp2J+C2wiwHsHj8XDBBRfgggsuYHvqX3jhBTz++OMoKytje+qjbbWkba9KpTIs1ZhEImGHKtDtrf7+fjgcDq8GlVgIMDo6CqPRGJLkc3NzuPHGG/Hwww/jsssui/p8geB0OnH99dfji1/8Iq677joAQH5+Pqanp1FYWIjp6WnWp/58vm8TUoxzuVzYsGED3nrrLRQXF2Pbtm344x//iM2bN0d3lecpqMHCCy+8gOPHjyM3Nxf79u3DNddcE7a1Mh1mkJeXF9TnPBw4nU5WoEP94qMR6IyNjWFhYSHkFqNer8d1112HBx98EHv27Inp2v2BEIJbb70VCoUCP//5z9nX//Ef/xFKpZItxul0Ojz66KNsfzstxt1zzz34+OOP435dfrA2q+4AcPz4cXzjG9+A2+3G7bffju9///vRHmpdgOq+Dx8+zLql7N27F9dccw1yc3P9Eo2OCk5Eb7uvQIfaZsnl8qCkHx8fh8FgCDm3fX5+Htdffz3uu+8+dqWNN/72t7/hkksu8bqWH//4x9ixYwcOHDiAiYkJlJeX4/nnn4dCoQAhBHfffTdef/11SKVSPP3009i6dWtCrs0Ha5foKUQPuh115MgRvPLKKxCLxdizZw/27duHgoICMAzDzkErLS2Ne7umLzweD3Q6HdRqNSvQyc/PX9aVNjExwc6QD0Zyo9GIG264AXfffTcOHjyY0GtfI0gRHVhTwoe4gxCCiYkJtr0WAC677DK88cYb+M1vfpP0Ti7frjQq0LFYLGFp6c1mMw4cOIDbb78dN998cxKvfFUjRXRgTQkfEgpCCDo6OrB3716Ul5fD6XTimmuuwb59+1BZWZn06jEV6AwNDbHuKvn5+cjJyfEr0LFarThw4ACrekuBxdoUzMQbKeHDIhiGwYkTJ/D000/jr3/9K15++WUolUp861vfwmc+8xk8+uij6O/vR4iHd1yvx2g0gsfj4dJLL0VVVRXMZrNfgY7NZsMXv/hF3HjjjbjtttuScn0pfIo1saJzMTY2hl27dqGrqwtlZWUwGAwAFleX7OxsGAwGXHPNNXjggQewc+dOAIuh7k9/+tNkFVZWBFqtFi+//DJefPFFzM7OevXUJ2qln5qaglqtRnNz87KmE2omqVKp8MMf/hAMw+Bzn/scHnzwwXWxbx0hUis6FynhQ2AolUrccccdOHbsGP73f/8XGzduxL/+679i586deOihh3DmzBl4PJ64nU+lUgUkOQDWrnnHjh3Iz89Hfn4+3n77bdx0000xnff2229HXl4eaxEGADqdDldccQVqa2txxRVXQK/XA1h8+N9zzz2oqalBU1MTTp8+HdO51zLWDNGDCR8ArBvhQziQy+W45ZZb8PLLL+Pdd99FW1sbfvazn+Hiiy/Ggw8+iJMnT8ZE+nPnzmF2djYgySlcLhfuvPNObN++HS+++CL+8pe/4Omnn476vADwpS99Ca+//rrXa4888gguu+wyDA4O4rLLLmM73V577TUMDg5icHAQ//3f/4277rorpnOvZawJohNCcMcdd6Curg7f+ta32Nf37t2LZ555BgDwzDPPYN++fezrv/vd70AIwYcffoisrKzzQq8cDTIyMnDo0CG88MILOHHiBC6++GL86le/wkUXXYT7778fH3zwAdxud9jHo4XRUCR3u9246667UF9fj+9+97tstBXrdNpdu3YtExOtt1pNVCCEBPtvVeC9994jAEhjYyNpbm4mzc3N5NixY2Rubo7s3r2b1NTUkMsuu4xotVpCCCEej4d87WtfI1VVVaShoYGcPHlyhd/B6oPVaiWvvvoqueWWW0hDQwP58pe/TF577TUyPz9PzGaz3/+GhobIu+++SxYWFgL+jNlsJgsLC+RLX/oSeeCBB4jH44n7tY+OjpLNmzezX2dlZbH/9ng87NdXX301ee+999jv7d69e7XeC6F4GPN/a6KpZefOnQEryW+99day1xiGwS9+8YtEX9aahkQiwZ49e7Bnzx44HA785S9/wZEjR3Dfffdhx44d2L9/Py655BK2p35mZoa1iA62kns8Hnz7299GdnY2Hn744aTXTdZ7rSYQ1gTRkw23242tW7eiuLgYR48eTcbY3BWFSCTCVVddhauuugoulwvvvvsuXnjhBXz3u99FW1sb8vPzYTQa8eijjwY1sPB4PHjggQcgEonw2GOPJc2fbj02qUSKNZGjJxtPPPEE6urq2K/vv/9+fPOb38TQ0BCys7Px1FNPAQCeeuopZGdnY2hoCN/85jdx//33r9Qlxw0CgQC7d+/Gk08+ibNnz2LDhg149tln8dFHH+ErX/kKXn31VVgslmW/5/F48IMf/AAOhwP/8R//kVQTylStJgyEiO3XHSYnJ8nu3bvJW2+9Ra6++mri8XiIUqkkTqeTEELIBx98QD772c8SQgj57Gc/Sz744ANCCCFOp5MolcqE5KQrBafTSW677TZiMBiI2+0mJ06cIN/61rdIU1MTuf7668nvf/97Mjs7S0wmE7n//vvJrbfeSlwuV0Kv6dChQ6SgoIAIBAJSXFxMfv3rX58PtZqE5+gpovvg+uuvJ5988gl5++23ydVXX000Gg2prq5mvz8xMcEWgjZv3kwmJyfZ71VVVRGNRpP0a0423G43+eSTT8j9999PWlpaSH19Pdm/f3/CSX4eI1WMSyaOHj2KvLw8bNmyBe+8885KX86qBY/Hw5YtW7Blyxb8+Mc/xtGjR7F79+7z3ZJ5TSNFdA7ef/99vPrqqzh+/DhsNhsWFhZw7733nhdjcxMFHo+HvXv3rvRlpBACqWIcBz/5yU8wNTWFsbExPPvss9i9ezf+8Ic/4DOf+QwOHz4MYHmxhxaBDh8+jN27d6e2dlJYlUgRPQz89Kc/xeOPP46amhpotVrccccdAIA77rgDWq0WNTU1ePzxx6MaMpBCCsnAmuteS2F94Hye9OMHqe618xkGgwE33HADNm3ahLq6Opw4cSLViYVFwdLXv/51vPbaa+jp6cGf/vQn9PT0rPRlrWmkiL6CuPfee3HVVVehr68PZ8+eRV1dXaoTC+tr0k+ykCL6CmF+fh7vvvsum++LRCLI5fJUJxYCT0xJIXqkiL5CGB0dRW5uLm677Ta0trbizjvvhNlsjnhcUAophIMU0VcILpcLp0+fxl133YX29nakp6cvq9qv106sVDNK/JEi+gqhpKQEJSUl2LFjBwDghhtuwOnTp1OuOQC2bduGwcFBjI6OwuFw4Nlnn02JcmJEiugrhIKCApSWlqK/vx/AYl99fX19qhMLix10//Vf/4Urr7wSdXV1OHDgQGqcV4xI7aOvIM6cOYM777wTDocDVVVVePrpp+HxeFbbuKAUEo/UAIcUUlgHSDjRQzW1rL9K0HkKhmG+CeBOLD68OwHcBqAQwLMAlABOAbiZEOJgGEYM4HcAtgDQAjhICBlbietOIT5I5ejrAAzDFAO4B8BWQkgDAD6AQwB+CuBnhJAaAHoAdyz9yh0A9Euv/2zp51JYw0gRff1AACCNYRgBACmAaQC7ARxe+v4zAPYv/Xvf0tdY+v5lzHrc5zuPkCL6OgAhRAXgMQATWCT4PBZDdQMhxLX0Y1MA6H5dMYDJpd91Lf38+mq0P8+QIvo6AMMw2VhcpSsBFAFIB3DVil5UCklFiujrA5cDGCWEaAghTgAvArgYgHwplAeAEgBUU6sCUAoAS9/PwmJRLoU1ihTR1wcmAFzAMIx0Kde+DEAPgLcB3LD0M7cCoC1iry59jaXv/4WE2IdNYXUj1D56CucJGIb5EYCDAFwA2rG41VaMxe01xdJr/x8hxM4wjATA7wG0AtABOEQIGVmRC08hLkgRPYUU1gFSoXsKKawDpIieQgrrACmip5DCOkCK6CmksA6QInoKKawDpIieQgrrACmip5DCOkCK6CmksA7w/wPWhJu78Lh/JAAAAABJRU5ErkJggg=="
  >
  </div>
  
  </div>
  
  </div>
  
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="Execute-wake-calculation">Execute wake calculation<a class="anchor-link" href="#Execute-wake-calculation">&#182;</a></h2><p>Running the wake calculation is a one-liner. This will calculate the velocities at each turbine given the wake of other turbines for every wind speed and wind direction combination.
  Since we have not explicitly specified yaw control settings, all turbines are aligned with the inflow.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
  <div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
       <div class="CodeMirror cm-s-jupyter">
  <div class=" highlight hl-ipython3"><pre><span></span><span class="n">fi</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">()</span>
  </pre></div>
  
       </div>
  </div>
  </div>
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="Get-turbine-power">Get turbine power<a class="anchor-link" href="#Get-turbine-power">&#182;</a></h2><p>At this point, the simulation has completed and we can use the <code>FlorisInterface</code> to extract useful information such as the power produced at each turbine. Remember that we have configured the simulation with two wind directions, two wind speeds, and four turbines.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
  <pre>Dimensions of `powers`
  (2, 2, 4)
  
  Turbine powers for 8 m/s
  Wind direction 0
  [1691.32648289 1691.32648289  592.65288889  592.97819946]
  Wind direction 1
  [1691.32648289 1691.32648289 1631.06709246 1629.75508349]
  
  Turbine powers for all turbines at all wind conditions
  [[[1691.32648289 1691.32648289  592.65288889  592.97819946]
    [2407.84258855 2407.84258855  861.30598083  861.73203268]]
  
   [[1691.32648289 1691.32648289 1631.06709246 1629.75508349]
    [2407.84258855 2407.84258855 2321.41264704 2319.53233514]]]
  </pre>
  </div>
  </div>
  
  </div>
  
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
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
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
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
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
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
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
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
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
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
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="Visualization">Visualization<a class="anchor-link" href="#Visualization">&#182;</a></h2><p>While comparing turbine and farm powers is meaningful, a picture is worth at least 1000 Watts, and the <code>FlorisInterface</code> provides powerful routines for visualization.</p>
  <p><strong>NOTE <code>floris.tools</code> is under active design and development. The API's will change and additional functionality from FLORIS v2 will be included in upcoming releases.</strong></p>
  <p>The visualization functions require that the user select a single atmospheric condition to plot. However, the internal data structures still have the same shape but the wind speed and wind direction
  dimensions have a size of 1. This means that the yaw angle array used for plotting must have the same shape as before but a single atmospheric condition must be selected.</p>
  <p>Let's create a horizontal slice of each atmospheric condition from above with and without yaw settings included. Notice that although we are plotting the conditions for two different wind directions,
  the farm is rotated so that the wind is coming from the left (west) in both cases.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
  <div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
       <div class="CodeMirror cm-s-jupyter">
  <div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">floris.tools.visualization</span> <span class="kn">import</span> <span class="n">visualize_cut_plane</span>
  <span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
  
  <span class="n">fig</span><span class="p">,</span> <span class="n">axarr</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
  
  <span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">210</span><span class="p">]</span> <span class="p">)</span>
  <span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;210 - Aligned&quot;</span><span class="p">)</span>
  
  <span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">210</span><span class="p">],</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">]</span> <span class="p">)</span>
  <span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;210 - Yawed&quot;</span><span class="p">)</span>
  
  <span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">270</span><span class="p">]</span> <span class="p">)</span>
  <span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;270 - Aligned&quot;</span><span class="p">)</span>
  
  <span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">270</span><span class="p">],</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span><span class="p">[</span><span class="mi">1</span><span class="p">:</span><span class="mi">2</span><span class="p">,</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">]</span> <span class="p">)</span>
  <span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;270 - Yawed&quot;</span><span class="p">)</span>
  </pre></div>
  
       </div>
  </div>
  </div>
  </div>
  
  <div class="jp-Cell-outputWrapper">
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>
  
  
  
  
  <div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
  <pre>&lt;matplotlib.collections.QuadMesh at 0x13eac2440&gt;</pre>
  </div>
  
  </div>
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  
  
  <div class="jp-RenderedImage jp-OutputArea-output ">
  <img src="data:image/png;base64,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"
  >
  </div>
  
  </div>
  
  </div>
  
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <p>We can also create a visualization of the streamwise inflow velocities on the turbine rotor grid points located on the rotor plane.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>
  
  
  
  
  <div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
  <pre>Text(0.5, 0.98, &#39;Wind direction 270&#39;)</pre>
  </div>
  
  </div>
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  
  
  <div class="jp-RenderedImage jp-OutputArea-output ">
  <img src="data:image/png;base64,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"
  >
  </div>
  
  </div>
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  
  
  <div class="jp-RenderedImage jp-OutputArea-output ">
  <img src="data:image/png;base64,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"
  >
  </div>
  
  </div>
  
  </div>
  
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="On-Grid-Points">On Grid Points<a class="anchor-link" href="#On-Grid-Points">&#182;</a></h2>
  </div>
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <p>In FLORIS, grid points are the points in space where the wind conditions are calculated.
  In a typical simulation, these are all located on a regular grid on each turbine rotor.</p>
  <p>The parameter <code>turbine_grid_points</code> specifies the number of rows and columns which define the turbine grid.
  In the example inputs, this value is 3 meaning there are 3 x 3 = 9 total grid points for each turbine.
  Wake steering codes currently require greater values greater than 1 in order to compute gradients.
  However, it is likely that a single grid point (1 x 1) is suitable for non wind farm control applications,
  although retuning of some parameters could be warranted.</p>
  <p>We can visualize the locations of the grid points in the current example using pyplot</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
  <pre>xs has shape:  (2, 1, 4, 3, 3)  of 2 wd x 2 ws x 4 turbines x 3 x 3 grid points
  </pre>
  </div>
  </div>
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>
  
  
  
  
  <div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
  <pre>(0.0, 150.0)</pre>
  </div>
  
  </div>
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  
  
  <div class="jp-RenderedImage jp-OutputArea-output ">
  <img src="data:image/png;base64,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"
  >
  </div>
  
  </div>
  
  </div>
  
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h1 id="Advanced-Usage-and-Concepts">Advanced Usage and Concepts<a class="anchor-link" href="#Advanced-Usage-and-Concepts">&#182;</a></h1>
  </div>
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="Calculating-AEP">Calculating AEP<a class="anchor-link" href="#Calculating-AEP">&#182;</a></h2>
  </div>
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <p>Calculating AEP in FLORIS V3 takes advantage of the new vectorized framework to substantially reduce the computation time with respect to V2.4.</p>
  <p>In these examples we demonstrate a simplied AEP calculation for a 25-turbine farm using several different modeling options.</p>
  <p>We will make a simplifying assumption that every wind speed and direction is equally likely.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
  <pre>Calculating AEP for 1440 wind direction and speed combinations...
  </pre>
  </div>
  </div>
  
  </div>
  
  </div>
  
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
  <pre>Number of turbines = 25
  </pre>
  </div>
  </div>
  
  </div>
  
  </div>
  
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <p>Calculate the AEP and use the jupyter time command to show computation time:</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
  <pre>CPU times: user 4.17 s, sys: 1.67 s, total: 5.84 s
  Wall time: 4.9 s
  </pre>
  </div>
  </div>
  
  </div>
  
  </div>
  
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
  <pre>CPU times: user 6.57 s, sys: 2.55 s, total: 9.12 s
  Wall time: 7.94 s
  </pre>
  </div>
  </div>
  
  </div>
  
  </div>
  
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  
  <div class="jp-OutputArea jp-Cell-outputArea">
  
  <div class="jp-OutputArea-child">
  
      
      <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
  
  
  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
  <pre>Jensen 843.1 GWh
  GCH 843.8 GWh
  </pre>
  </div>
  </div>
  
  </div>
  
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <h2 id="Wake-Steering-Design">Wake Steering Design<a class="anchor-link" href="#Wake-Steering-Design">&#182;</a></h2>
  </div>
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <p>FLORIS V3 further includes new optimization routines for the design of wake steering controllers.  The SerialRefine is a new method for quickly identifying optimum yaw angles.</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
  <div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
       <div class="CodeMirror cm-s-jupyter">
  <div class=" highlight hl-ipython3"><pre><span></span><span class="o">%%time</span>
  <span class="c1">## Calculate the optimum yaw angles for 25 turbines and 72 wind directions</span>
  <span class="n">df_opt</span> <span class="o">=</span> <span class="n">yaw_opt</span><span class="o">.</span><span class="n">_optimize</span><span class="p">()</span>
  </pre></div>
  
       </div>
  </div>
  </div>
  </div>
  
  </div>
  <div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
  </div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
  <p>In the results, T0 is the upstream turbine when wind direction is 270, while T6 is upstream at 90 deg</p>
  
  </div>
  </div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
  <div class="jp-Cell-inputWrapper">
  <div class="jp-InputArea jp-Cell-inputArea">
  <div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
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
  
  </div>
  </body>
  
  
  </html>
  