---
layout: default
title: Overview
nav_order: 1
---
<html>
<head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>00_getting_started</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>




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



<style type="text/css">
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
  align-items: flex-end;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
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


.lm-TabBar-addButton.lm-mod-hidden {
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

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
  background: inherit;
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
  padding-right:5px;
  z-index:30; }
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
.bp3-multistep-dialog-panels{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }

.bp3-multistep-dialog-left-panel{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column; }
  .bp3-dark .bp3-multistep-dialog-left-panel{
    background:#202b33; }

.bp3-multistep-dialog-right-panel{
  background-color:#f5f8fa;
  border-left:1px solid rgba(16, 22, 26, 0.15);
  border-radius:0 0 6px 0;
  -webkit-box-flex:3;
      -ms-flex:3;
          flex:3;
  min-width:0; }
  .bp3-dark .bp3-multistep-dialog-right-panel{
    background-color:#293742;
    border-left:1px solid rgba(16, 22, 26, 0.4); }

.bp3-multistep-dialog-footer{
  background-color:#ffffff;
  border-radius:0 0 6px 0;
  border-top:1px solid rgba(16, 22, 26, 0.15);
  padding:10px; }
  .bp3-dark .bp3-multistep-dialog-footer{
    background:#30404d;
    border-top:1px solid rgba(16, 22, 26, 0.4); }

.bp3-dialog-step-container{
  background-color:#f5f8fa;
  border-bottom:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-dialog-step-container{
    background:#293742;
    border-bottom:1px solid rgba(16, 22, 26, 0.4); }
  .bp3-dialog-step-container.bp3-dialog-step-viewed{
    background-color:#ffffff; }
    .bp3-dark .bp3-dialog-step-container.bp3-dialog-step-viewed{
      background:#30404d; }

.bp3-dialog-step{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#f5f8fa;
  border-radius:6px;
  cursor:not-allowed;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:4px;
  padding:6px 14px; }
  .bp3-dark .bp3-dialog-step{
    background:#293742; }
  .bp3-dialog-step-viewed .bp3-dialog-step{
    background-color:#ffffff;
    cursor:pointer; }
    .bp3-dark .bp3-dialog-step-viewed .bp3-dialog-step{
      background:#30404d; }
  .bp3-dialog-step:hover{
    background-color:#f5f8fa; }
    .bp3-dark .bp3-dialog-step:hover{
      background:#293742; }

.bp3-dialog-step-icon{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:rgba(92, 112, 128, 0.6);
  border-radius:50%;
  color:#ffffff;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:25px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:25px; }
  .bp3-dark .bp3-dialog-step-icon{
    background-color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#2b95d6; }
  .bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#8a9ba8; }

.bp3-dialog-step-title{
  color:rgba(92, 112, 128, 0.6);
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  padding-left:10px; }
  .bp3-dark .bp3-dialog-step-title{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-title{
    color:#2b95d6; }
  .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
    color:#182026; }
    .bp3-dark .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
      color:#f5f8fa; }
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
  .bp3-control-group .bp3-input-group > .bp3-input-left-container,
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
  table.bp3-html-table tbody tr:first-child td,
  .bp3-running-text table tfoot tr:first-child th,
  table.bp3-html-table tfoot tr:first-child th,
  .bp3-running-text table tfoot tr:first-child td,
  table.bp3-html-table tfoot tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
  .bp3-dark .bp3-running-text table tbody tr:first-child td,
  .bp3-running-text .bp3-dark table tbody tr:first-child td,
  .bp3-dark table.bp3-html-table tbody tr:first-child td,
  .bp3-dark .bp3-running-text table tfoot tr:first-child th,
  .bp3-running-text .bp3-dark table tfoot tr:first-child th,
  .bp3-dark table.bp3-html-table tfoot tr:first-child th,
  .bp3-dark .bp3-running-text table tfoot tr:first-child td,
  .bp3-running-text .bp3-dark table tfoot tr:first-child td,
  .bp3-dark table.bp3-html-table tfoot tr:first-child td{
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

table.bp3-html-table.bp3-html-table-bordered tbody tr td,
table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
  table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
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
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td,
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
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
.bp3-panel-stack2{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack2-header{
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
  .bp3-dark .bp3-panel-stack2-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack2-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack2-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack2-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack2-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack2-view{
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
  .bp3-dark .bp3-panel-stack2-view{
    background-color:#30404d; }
  .bp3-panel-stack2-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter, .bp3-panel-stack2-push .bp3-panel-stack2-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter-active, .bp3-panel-stack2-push .bp3-panel-stack2-appear-active{
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

.bp3-panel-stack2-push .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-push .bp3-panel-stack2-exit-active{
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

.bp3-panel-stack2-pop .bp3-panel-stack2-enter, .bp3-panel-stack2-pop .bp3-panel-stack2-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack2-pop .bp3-panel-stack2-enter-active, .bp3-panel-stack2-pop .bp3-panel-stack2-appear-active{
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

.bp3-panel-stack2-pop .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-pop .bp3-panel-stack2-exit-active{
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
  right:0;
  z-index:40; }
  .bp3-toast-container.bp3-toast-container-in-portal{
    position:fixed; }
  .bp3-toast-container.bp3-toast-container-inline{
    position:absolute; }
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
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
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
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
  --jp-icon-listings-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MC45NzggNTAuOTc4IiBzdHlsZT0iZW5hYmxlLWJhY2tncm91bmQ6bmV3IDAgMCA1MC45NzggNTAuOTc4OyIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+Cgk8Zz4KCQk8cGF0aCBzdHlsZT0iZmlsbDojMDEwMDAyOyIgZD0iTTQzLjUyLDcuNDU4QzM4LjcxMSwyLjY0OCwzMi4zMDcsMCwyNS40ODksMEMxOC42NywwLDEyLjI2NiwyLjY0OCw3LjQ1OCw3LjQ1OAoJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDAKCQkJYzYuODE2LDAsMTMuMjIxLTIuNjQ4LDE4LjAyOS03LjQ1OGM0LjgwOS00LjgwOSw3LjQ1Ny0xMS4yMTIsNy40NTctMTguMDNDNTAuOTc3LDE4LjY3LDQ4LjMyOCwxMi4yNjYsNDMuNTIsNy40NTh6CgkJCSBNNDIuMTA2LDQyLjEwNWMtNC40MzIsNC40MzEtMTAuMzMyLDYuODcyLTE2LjYxNSw2Ljg3MmgtMC4wMDJjLTYuMjg1LTAuMDAxLTEyLjE4Ny0yLjQ0MS0xNi42MTctNi44NzIKCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzIKCQkJYzQuNDMxLDQuNDMxLDYuODcxLDEwLjMzMiw2Ljg3MSwxNi42MTdDNDguOTc3LDMxLjc3Miw0Ni41MzYsMzcuNjc1LDQyLjEwNiw0Mi4xMDV6Ii8+CgkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik0yMy41NzgsMzIuMjE4Yy0wLjAyMy0xLjczNCwwLjE0My0zLjA1OSwwLjQ5Ni0zLjk3MmMwLjM1My0wLjkxMywxLjExLTEuOTk3LDIuMjcyLTMuMjUzCgkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUKCQkJYzAtMS4wOTYtMC4yNi0yLjA4OC0wLjc3OS0yLjk3OWMtMC41NjUtMC44NzktMS41MDEtMS4zMzYtMi44MDYtMS4zNjljLTEuODAyLDAuMDU3LTIuOTg1LDAuNjY3LTMuNTUsMS44MzIKCQkJYy0wLjMwMSwwLjUzNS0wLjUwMywxLjE0MS0wLjYwNywxLjgxNGMtMC4xMzksMC43MDctMC4yMDcsMS40MzItMC4yMDcsMi4xNzRoLTIuOTM3Yy0wLjA5MS0yLjIwOCwwLjQwNy00LjExNCwxLjQ5My01LjcxOQoJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQoJCQljMCwxLjE0Mi0wLjEzNywyLjExMS0wLjQxLDIuOTExYy0wLjMwOSwwLjg0NS0wLjczMSwxLjU5My0xLjI2OCwyLjI0M2MtMC40OTIsMC42NS0xLjA2OCwxLjMxOC0xLjczLDIuMDAyCgkJCWMtMC42NSwwLjY5Ny0xLjMxMywxLjQ3OS0xLjk4NywyLjM0NmMtMC4yMzksMC4zNzctMC40MjksMC43NzctMC41NjUsMS4xOTljLTAuMTYsMC45NTktMC4yMTcsMS45NTEtMC4xNzEsMi45NzkKCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+Cgk8L2c+Cjwvc3ZnPgo=);
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
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
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
.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
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
.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
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

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px 5px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
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
  color: var(--jp-ui-font-color2);
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
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
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
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
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

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
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

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FileDialog-Checkbox {
  margin-top: 35px;
  display: flex;
  flex-direction: row;
  align-items: end;
  width: 100%;
}

.jp-FileDialog-Checkbox > label {
  flex: 1 1 auto;
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
  outline: none;
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

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .remote-caret {
  position: relative;
  border-left: 2px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .remote-caret > div {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -2px;
  font-size: 0.95em;
  background-color: rgb(250, 129, 0);
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 3;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .remote-caret.hide-name > div {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .remote-caret:hover > div {
  opacity: 1;
  transition-delay: 0s;
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
  .jp-ToolbarButtonComponent:focus-visible {
  background-color: var(--jp-brand-color0);
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
  color: var(--jp-error-color1);
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

.jp-DirListing:focus-visible {
  border: 1px solid var(--jp-brand-color1);
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

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
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
  color: var(--jp-ui-inverse-font-color1);
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
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon:before {
  color: var(--jp-ui-inverse-font-color1);
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

body[data-format='mobile'] .jp-OutputArea-child {
  flex-direction: column;
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

body[data-format='mobile'] .jp-OutputPrompt {
  flex: 0 0 auto;
  text-align: left;
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

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  margin-left: var(--jp-notebook-padding);
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

body[data-format='mobile'] .jp-InputArea {
  flex-direction: column;
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

body[data-format='mobile'] .jp-InputArea-editor {
  margin-left: var(--jp-notebook-padding);
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

body[data-format='mobile'] .jp-InputPrompt {
  flex: 0 0 auto;
  text-align: left;
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

.jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
}

.jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-collapseHeadingButton {
  display: none;
}

.jp-MarkdownCell:hover .jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
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

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook-render * {
  contain: none !important;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
  float: left;
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

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt:before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
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
  display: block;
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

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body div {
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>

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

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);

  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
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

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

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
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
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

.CodeMirror pre {
  margin: 0;
  padding: 0;
}

/* Using table instead of flexbox so that we can use break-inside property */
/* CSS rules under this comment should not be required anymore after we move to the JupyterLab 4.0 CSS */


.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  min-width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-OutputArea-child {
  display: table;
  width: 100%;
}

.jp-OutputPrompt {
  display: table-cell;
  vertical-align: top;
  min-width: var(--jp-cell-prompt-width);
}

body[data-format='mobile'] .jp-OutputPrompt {
  display: table-row;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  display: table-row;
}

.jp-OutputArea-output.jp-OutputArea-executeResult {
  width: 100%;
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }

  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}
</style>

<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: { 
                    automatic: true 
                    }
                }
            });
        
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --></head>
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
<ul>
<li>preprocess</li>
<li>calculation</li>
<li>postprocess</li>
</ul>
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
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>

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
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Dimensions of grid x-components&quot;</span><span class="p">)</span>
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

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Turbine powers for 8 m/s&quot;</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">2</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Wind direction </span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">powers</span><span class="p">[</span><span class="n">i</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="p">:])</span>


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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Yaw turbine 0 by +25 degrees</span>
<span class="n">yaw_angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span> <span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">4</span><span class="p">)</span> <span class="p">)</span>
<span class="n">yaw_angles</span><span class="p">[:,</span> <span class="p">:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">25</span>

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
<pre>[[[25.  0.  0.  0.]
  [25.  0.  0.  0.]]

 [[25.  0.  0.  0.]
  [25.  0.  0.  0.]]]
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
<ul>
<li>Load an input file</li>
<li>Modify the inputs with a more complex wind turbine layout</li>
<li>Change the wind speeds and wind directions</li>
<li>Execute the simulation</li>
<li>Get the total farm power</li>
<li>Add yaw settings for some turbines</li>
<li>Execute the simulation</li>
<li>Get the total farm power and compare to without yaw control</li>
</ul>

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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fi</span> <span class="o">=</span> <span class="n">FlorisInterface</span><span class="p">(</span><span class="s2">&quot;inputs/gch.yaml&quot;</span><span class="p">)</span>

<span class="c1"># Construct the model</span>
<span class="n">D</span> <span class="o">=</span> <span class="mf">126.0</span>  <span class="c1"># Design the layout based on turbine diameter</span>
<span class="n">x</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span>  <span class="mi">6</span> <span class="o">*</span> <span class="n">D</span><span class="p">,</span> <span class="mi">6</span> <span class="o">*</span> <span class="n">D</span><span class="p">]</span>
<span class="n">y</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span> <span class="o">*</span> <span class="n">D</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">3</span> <span class="o">*</span> <span class="n">D</span><span class="p">]</span>
<span class="n">wind_directions</span> <span class="o">=</span> <span class="p">[</span><span class="mf">270.0</span><span class="p">,</span> <span class="mf">265.0</span><span class="p">]</span>
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
<span class="n">yaw_angles</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">25</span>            <span class="c1"># At 270 degrees, yaw the first turbine 25 degrees</span>
<span class="n">yaw_angles</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="p">:,</span> <span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="mi">25</span>           <span class="c1"># At 270 degrees, yaw the second turbine 25 degrees</span>
<span class="n">yaw_angles</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="p">:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="o">-</span><span class="mi">25</span>            <span class="c1"># At 265 degrees, yaw the first turbine -25 degrees</span>
<span class="n">yaw_angles</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="p">:,</span> <span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="o">-</span><span class="mi">25</span>            <span class="c1"># At 265 degrees, yaw the second turbine -25 degrees</span>

<span class="c1"># Calculate the velocities at each turbine for all atmospheric conditions given the yaw control settings</span>
<span class="n">fi</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">(</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span> <span class="p">)</span>

<span class="c1"># Get the farm power</span>
<span class="n">turbine_powers</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">get_turbine_powers</span><span class="p">()</span> <span class="o">/</span> <span class="mf">1000.0</span>
<span class="n">farm_power_yaw</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">turbine_powers</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>

<span class="c1"># Compare power difference with yaw</span>
<span class="n">difference</span> <span class="o">=</span> <span class="mi">100</span> <span class="o">*</span> <span class="p">(</span><span class="n">farm_power_yaw</span> <span class="o">-</span> <span class="n">farm_power_baseline</span><span class="p">)</span> <span class="o">/</span> <span class="n">farm_power_baseline</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Power </span><span class="si">% d</span><span class="s2">ifference with yaw&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;    270 degrees: </span><span class="si">{</span><span class="n">difference</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]</span><span class="si">:</span><span class="s2">4.2f</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;    265 degrees: </span><span class="si">{</span><span class="n">difference</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">]</span><span class="si">:</span><span class="s2">4.2f</span><span class="si">}</span><span class="s2">%&quot;</span><span class="p">)</span>
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
    270 degrees: 7.39%
    265 degrees: 7.17%
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

<span class="n">fig</span><span class="p">,</span> <span class="n">axarr</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">270</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;270 - Aligned&quot;</span><span class="p">)</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">270</span><span class="p">],</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;270 - Yawed&quot;</span><span class="p">)</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">265</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;265 - Aligned&quot;</span><span class="p">)</span>

<span class="n">horizontal_plane</span> <span class="o">=</span> <span class="n">fi</span><span class="o">.</span><span class="n">calculate_horizontal_plane</span><span class="p">(</span> <span class="n">wd</span><span class="o">=</span><span class="p">[</span><span class="mi">265</span><span class="p">],</span> <span class="n">yaw_angles</span><span class="o">=</span><span class="n">yaw_angles</span><span class="p">[</span><span class="mi">1</span><span class="p">:</span><span class="mi">2</span><span class="p">,</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">]</span> <span class="p">,</span> <span class="n">height</span><span class="o">=</span><span class="mf">90.0</span> <span class="p">)</span>
<span class="n">visualize_cut_plane</span><span class="p">(</span><span class="n">horizontal_plane</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axarr</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;265 - Yawed&quot;</span><span class="p">)</span>
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
<pre>&lt;matplotlib.collections.QuadMesh at 0x1691757c0&gt;</pre>
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
<span class="n">fig</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s2">&quot;Wind direction 270&quot;</span><span class="p">)</span>

<span class="n">fig</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">plot_rotor_values</span><span class="p">(</span><span class="n">fi</span><span class="o">.</span><span class="n">floris</span><span class="o">.</span><span class="n">flow_field</span><span class="o">.</span><span class="n">u</span><span class="p">,</span> <span class="n">wd_index</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">ws_index</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">n_rows</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">n_cols</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">return_fig_objects</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">fig</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s2">&quot;Wind direction 265&quot;</span><span class="p">)</span>
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
<pre>Text(0.5, 0.98, &#39;Wind direction 265&#39;)</pre>
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
<img src="data:image/png;base64,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cHAQTz/9NO66665V/gSxgUKhwK233opXXnkFLS0tyMjIgFqtxvnnn49HHnkEx48fh8fjWbXzm5qaYt1zKMk7Oztx991348CBA2wHn0gkipjkX/nKV5CXl4ft27ezj+n1elx22WWorq7GZZddBoPBAGBpW3PvvfeiqqoKDQ0NOHnyZHQfcB1iXRF906ZN7IqckZGB2tpaTE1N4dVXX8Wtt94KACwBAODVV1/Fl770JTAMg7PPPhtGoxEzMzOrdfoxh9PpxIUXXoi2tjYcPHgQH330Ec4++2w89dRTOOecc/D1r38dR44cgdvtTtg5TU9Ps7bVNLro6enBHXfcgRdeeAFVVVUxeZ8vf/nLeOONN7we22g3/LBACAn235rF6OgoKSkpISaTicjlcvZxj8fD/nz11VeT999/n/3dJZdcQj755JMEn+nqwGq1kldeeYXccsstZNu2beTOO+8kf/3rX4nJZCJmszku/w0NDZH33nuPzM/Ps4+dPHmS1NfXk9OnT8f8M46OjpJt27axP2/ZsoVMT08TQgiZnp4mW7ZsIYQQcscdd5A///nPfp+3RhCKh1H/t65WdIqNLGXkC5lMhj179uAPf/gDTpw4gd27d+PFF1/EOeecg3vvvRdvv/02nE5nzN5vdnYWU1NTXiq8oaEh3HrrrfjDH/6QkJJquNr1jYR1R/RgUkYAZ7yUMRJIpVJcddVVeOaZZ9De3o4vfOELaGtrw7nnnou77roLb775Jux2e8THn5ubw8TEhBfJx8fH8cUvfhHPPvssGhsbY/VReCN5w/fGuiI6SUoZo4ZYLMZll12GX/3qV+jo6MBtt92Gt956CxdccAG++tWvoq2tDVarlffx1Go1VCqVV2fc5OQkbrrpJjz99NNobW2N10dZgeQNPwhCxPZrCu+//z4BQOrr60ljYyNpbGwkhw4dIlqtllxyySWkqqqKXHrppUSn0xFClvbr//qv/0oqKirI9u3bN8z+PBK4XC7ywQcfkPvvv5/U19eT66+/nvzxj38karU64J58bGyMvPPOO8RoNHrt05uamsh7770X93P23aP/n//zf8j3v/99Qggh3//+98mDDz5ICCGkra2NXHHFFcTj8ZAjR46QnTt3xv3cwkTc9+jriuhJJAZut5scO3aMPPjgg6SxsZHs3buXPPPMM2RmZiYoyYeHh0lLSwt5++23436O+/btIwUFBUQkEpGioiLym9/8Zj3f8ONO9HUlmFkNuN1u7NixA0VFRWhra0vErOw1BToxZf/+/Th8+DA2bdqE2tpajIyM4JlnnmFbUzUaDa677jp873vfw+WXX77KZ73ukBTMrDaefPJJ1NbWsj8/9NBDeOCBBzA0NASFQoHf/va3AIDf/va3UCgUGBoawgMPPICHHnpotU45phAIBGhqasJ//dd/4cSJE7j++uvx/PPPY2ZmBl/4whfw3HPPYXBwEDfccAO+853vJEm+VhFiyd/QmJiYIJdccgl56623yNVXX008Hg/Jzs4mTqeTEELIRx99RC6//HJCCCGXX345+eijjwghhDidTpKdnU08Hs+qnXu8cP/995PZ2Vni8XhIf38/+e53v0sKCwvJr3/969U+tfWM5B59NXHdddeR48ePk3feeYdcffXVRKPRkMrKSvb3KpWKTQZt27aNTExMsL+rqKggGo0m4ee8GjgTb2gJRtyJngzdA6CtrQ15eXkJLQ+tV8S7Xv3GG29g69atqKqqYmWtSYSHM262Tazw4Ycf4rXXXsPhw4dhs9kwPz+P++6774yYlb2esFGchOON5IoeAN///vcxOTmJsbExPP/887jkkkvwpz/9CRdffDH2798PYKU4h4p29u/fj0suuSSpzIoBuE7CEomEdRJOIjwkiR4mfvCDH+CJJ55AVVUVdDodbr/9dgDA7bffDp1Oh6qqKjzxxBPJEDNGSOrUY4Nk6M4DF110ES666CIAQEVFBY4dO7biOTKZDC+99FKCzyyJJPghuaKvERiNRlx//fWoqalBbW0tjhw5kjRSQFKnHiskib5GcN999+GKK65AX18fOjo6UFtbmzRSALBz504MDg5idHQUDocDzz//PHbv3r3ap7X+EKL+lkQCYDQaSVlZ2Yp69Do2UogpDh06RKqrq0lFRQX5r//6r9U+nXgg7nX05B59DWB0dBS5ubm47bbb0NHRgdbWVjz55JMbaghgMFx11VW46qqrVvs01jWSofsagMvlwsmTJ3HXXXfh1KlTSEtLW5G1TxopJBENkkRfAyguLkZxcTF27doFALj++utx8uTJpJFCEjFDkuhrAAUFBSgpKUF/fz+ApZlkdXV1SeecJGKGUP3oSSQIDMM0AfgNAAmAEQC3YelG/CKAUgDjAG4khOiZpRj+5wCuAGABcBsh5PhqnHcS6wNJoieRxAZAMnTfwGAY5gGGYboZhjnNMMxfGIaRMQxTzjDMUYZhhhiGeYFhGMnyc6XLPw8t/75slU8/iTCQJPoGBcMwRQDuBbCDELIdgBDAPgA/APATQkgVAAOA25dfcjsAw/LjP1l+XhLrBEmib2yIAKQwDCMCkApgBsAlAPYv//53APYu/3vP8s9Y/v2lTLLet26QJPoGBSFkCsCPAKiwRHATgBMAjIQQ1/LTJgHQul0RgInl17qWn59suF8nSBJ9g4JhGAWWVulyAIUA0rCUxU/iDESS6BsXnwEwSgjREEKcAA4COA9A1nIoDwDFAGjz9xSAEgBY/r0cgC6xp5xEpEgSfeNCBeBshmFSl/falwLoAfAOgOuXn3MrAGrn8tryz1j+/dskWZtdN0jW0TcwGIb5DoAvAHABOAXgq1jaiz8PQLn82C2EEDvDMDIAfwDQDEAPYB8hZGRVTjyJsJEkehJJbAAkQ/ckktgASBI9iSQ2AJJETyKJDYAk0ZNIYgMgSfQkktgASBI9iSQ2AJJETyKJDYAk0ZNIYgPg/wflLU4NkyCSXQAAAABJRU5ErkJggg=="
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
<pre>CPU times: user 3.58 s, sys: 1.35 s, total: 4.94 s
Wall time: 3.91 s
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
<pre>CPU times: user 6.35 s, sys: 1.58 s, total: 7.93 s
Wall time: 6.71 s
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
<span class="n">fi_cc</span><span class="o">.</span><span class="n">calculate_wake</span><span class="p">()</span>
<span class="n">cc_aep</span> <span class="o">=</span> <span class="n">fi_cc</span><span class="o">.</span><span class="n">get_farm_power</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">/</span> <span class="n">num_bins</span>  <span class="o">/</span> <span class="mf">1E9</span> <span class="o">*</span> <span class="mi">365</span> <span class="o">*</span> <span class="mi">24</span>
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
<pre>CPU times: user 8.71 s, sys: 1.48 s, total: 10.2 s
Wall time: 10.3 s
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Show the results</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Jensen </span><span class="si">%.1f</span><span class="s1"> GWh&#39;</span> <span class="o">%</span> <span class="n">jensen_aep</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;GCH </span><span class="si">%.1f</span><span class="s1"> GWh&#39;</span> <span class="o">%</span> <span class="n">gch_aep</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;CC </span><span class="si">%.1f</span><span class="s1"> GWh&#39;</span> <span class="o">%</span> <span class="n">cc_aep</span><span class="p">)</span>
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
CC 839.3 GWh
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
CPU times: user 2.97 s, sys: 398 ms, total: 3.37 s
Wall time: 3.05 s
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
<img src="data:image/png;base64,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xktEREQkImrxEhEREYmICi8RERGRiKjwEhEREYmICi8RERGRiKjwEhEREYmICi8RERGRiKjwEhEREYlI32wHkIqhQ4d6RUVFqMdobm5m4MCBoR4jVyk3wZSbYMpNMOUmMeUlmHITrCfmZunSpR8E3SQ7JwqviooKlixZEuoxamtrqa6uDvUYuUq5CabcBFNugik3iSkvwZSbYD0xN2a2MWiduhpFREREIqLCS0RERCQiKrxEREREIpITY7xERESkd2ptbaWuro6Wlpa0th88eDCrV6/OcFSpKS4uZtSoURQWFqa8jQovERERyZq6ujpKS0upqKjAzLq8fWNjI6WlpSFElpy7U19fT11dHZWVlSlvp65GERERyZqWlhbKysrSKrqyycwoKyvrckudCi8RERHJqlwrutqlE3dKXY1mNgQYAewANrj7ni4fSaSHeumllzjyyCPp379/tkMRkQgtW7aMww8/PCvdVNIz1NfXc+qppwKwefNmCgoKGDZsGC0tLQwYMIC2tjZ2797NxRdfzA9+8IOMHDOw8DKzwcBXgcuBfsAWoBgoN7MXgNvcfWFGohDJkh07dnD88cfz9a9/nX/7t3/LdjgiEhF354QTTuDKK6/k9ttvz3Y4kiVlZWUsX74cgDlz5lBSUsINN9yAu9Pc3ExJSQmtra2ccMIJnHXWWRxzzDHdPmayrsYHgbeBE919rLuf4O5T3P0g4BbgPDO7LmhjMzvIzBaa2SozW2lm18eX729mT5jZ2vjPId3+FCJp2rZtG62trdx11120trZmOxwRicj27dvZvn07CxYsYPv27dkOR3oYM6OkpASIXXXZ2tqase7QwBYvd5+RZN1SYGkn+94NfNPdXzazUmCpmT0B1ABPuvstZnYjcCPwnS5HLpIBDQ0NAGzZsoXHHnuMc889N8sRiUgU2r/7jY2N/Pa3v+XKK6/MckQCMHv27L0tUKlqa2ujoKAgcH1VVRW33nprl2Npa2tj8uTJrFu3jq9+9atMmzaty/tIpNPB9WY2KcHjMDNLOj7M3Te5+8vx543AamAkcB4wP/62+cD53foEIt3Q/ssXYO7cuVmMRESipO++dKagoIDly5dTV1fHiy++yIoVKzKy31QG198GTAJeBQw4ElgJDDazr7j7nzrbgZlVABOBxUC5u2+Kr9oMlKcRt0hGtP/ynTJlCo8++ihb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y81sQtRxSH7LRFcjwOTJkxkxYoS6HERyRLr3adzXcccdx+jRo/Xdly7rm4VjTgXWuft6ADNbAJwHrMpCLABs376dxsbGvX3/8km9MTfbtm3j0EMP7fZ+CgoKuPrqq/npT3/K2rVrGTp0aAai6x1643mTKcpNYlHkZcuWLd0a39XOzKipqeGmm27ilVde4eCDD85AdMF0zgTram5KS0vp2zcb5U+cu0f6AC4G/rvD66uBXyTbZvLkyR6m66+/3gE98uwxc+bMbp87Cxcu9NWrV2f9s+ihhx6pP84777xuf/fd3d966y03s6x/Hj269li8eHFG/v2TAZZ4QE2TxZIvOTObBcwCKC8vD/WmpBUVFXzpS1+iX79+oR0jl+3atatX5ubYY4/t9nnV1NTE5s2b+eEPf8h7772XmcB6id563mSCcpNYVHmZOnVqxv5P+fGPf8w777yTkX0lo3MmWFdzU1dXx/bt2ZuswTziyR/N7FhgjrufEX/9XQB3/0nQNlOmTPElS5aEGldtbS3V1dWhHiNXKTfBlJtgyk0w5SYx5SWYchOsJ+bGzJa6+5RE67JxVeNLwGgzqzSzfsRuRfRIFuIQERERiVTkLV4AZnY2cCtQANzh7j/u5P1bgI0hhzUU+CDkY+Qq5SaYchNMuQmm3CSmvARTboL1xNwc4u7DEq3ISuHVE5nZkqBmwXyn3ARTboIpN8GUm8SUl2DKTbBcy01WJlAVERERyUcqvEREREQiosLrY7dnO4AeTLkJptwEU26CKTeJKS/BlJtgOZUbjfESERERiYhavEREREQiosJLREREJCIqvEREREQiosJLREREJCIqvEREREQiosJLREREJCIqvEREREQi0jfbAaRi6NChXlFREeoxmpubGThwYKjHyFXKTTDlJphyE0y5SUx5CabcBOuJuVm6dOkHQTfJzonCq6KigiVLloR6jNraWqqrq0M9Rq5SboIpN8GUm2DKTWLKSzDlJlhPzI2ZbQxap65GyWvuzhe/+EWeffbZbIciIiJ5ICdavETC8tFHHzF37lyWLl3KsmXL6NNHf4uIiEh49L+M5LWmpiYAXn31VX73u99lNxgREen11OIlea298AKYM2cO559/vlq9REQi0traSl1dHS0tLWnvY/DgwaxevTqDUaWuuLiYUaNGUVhYmPI2Krwkr7UXXpdddhkLFizg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xIm8+OKL2Q5HRER6ORVektc++ugjGhoaWL16Ncceeyw33ngjbW1t2Q5LRER6KRVekteampoAuPXWW5k5cyb//M//zO9+97vsBiUiIr1WTozxEglLe+E1cuRI/vZv/5Z7772X5557josuuijLkYmI9H6tra3U1dXR0tKS9j4GDx7M6tWrMxhV6oqLixk1ahSFhYUpb6PCS/Jae+FVUlJCv379mDRpEosXL85yVCIi+aGuro7S0lIqKiows7T20djYSGlpaYYj65y7U19fT11dHZWVlSlvp65GyWsdCy+AadOmsXTpUlpbW7MZlohIXmhpaaGsrCztoiubzIyysrIut9ap8JK8lqjwamlp4bXXXstmWCIieSMXi6526cSeUuFlZkPM7AgzO9TMVKxJr5Go8ALU3Sgikgfq6+upqqqiqqqKAw88kJEjR+59/f7773PxxRczbtw4xo8fz/PPP5+RYwaO8TKzwcBXgcuBfsAWoBgoN7MXgNvcfWFGohDJkn0Lr4qKCg444AAWL17MV77ylWyGJiIiISsrK2P58uUAzJkzh5KSEm644QYAZs6cyZlnnsmDDz7Irl272L59e0aOmWxw/YPAncCJ7r6t4wozmwxcbWaHuvuvE21sZgfFty8HHLjd3f/dzPYH/geoADYAl7j7h938HCJp2bfwMjOmTZumFi8RkTz20Ucf8cwzzzBv3jwA+vXrl9bdTRIJLLzcfUaSdUuBpZ3sezfwTXd/2cxKgaVm9gRQAzzp7reY2Y3AjcB3uhy5SAY0NTVhZvTv33/vsmnTpvG///u/bNu2jf322y97wYmI5JHZs2fvbX3qira2NgoKChKuq6qq4tZbb+3yPt98802GDRvGtddeyyuvvMLkyZP593//94zcE7LT8VpmNinB4zAzSzoVhbtvcveX488bgdXASOA8YH78bfOB87v1CUS6oampiYEDB9Knz8dfhfZxXi+99FK2whIRkSzavXs3L7/8Ml/5yldYtmwZAwcO5JZbbsnIvlOZx+s2YBLwKmDAkcBKYLCZfcXd/9TZDsysApgILAbK3X1TfNVmYl2RIlnR1NS0t5ux3dFHH42ZsXjxYmbMCGz4FRGRDEqnZQrCmcdr1KhRjBo1au8f4hdffHGkhde7wHXuvhLAzCYAPwS+DTwEJC28zKwE+A0w290bOl566e5uZh6w3SxgFkB5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"
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
