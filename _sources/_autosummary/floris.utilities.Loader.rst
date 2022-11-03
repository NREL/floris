floris.utilities.Loader
=======================

.. currentmodule:: floris.utilities

.. autoclass:: Loader
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~Loader.add_constructor
      ~Loader.add_implicit_resolver
      ~Loader.add_indent
      ~Loader.add_multi_constructor
      ~Loader.add_path_resolver
      ~Loader.ascend_resolver
      ~Loader.check_block_entry
      ~Loader.check_data
      ~Loader.check_directive
      ~Loader.check_document_end
      ~Loader.check_document_start
      ~Loader.check_event
      ~Loader.check_key
      ~Loader.check_node
      ~Loader.check_plain
      ~Loader.check_printable
      ~Loader.check_resolver_prefix
      ~Loader.check_state_key
      ~Loader.check_token
      ~Loader.check_value
      ~Loader.compose_document
      ~Loader.compose_mapping_node
      ~Loader.compose_node
      ~Loader.compose_scalar_node
      ~Loader.compose_sequence_node
      ~Loader.construct_document
      ~Loader.construct_mapping
      ~Loader.construct_object
      ~Loader.construct_pairs
      ~Loader.construct_scalar
      ~Loader.construct_sequence
      ~Loader.construct_undefined
      ~Loader.construct_yaml_binary
      ~Loader.construct_yaml_bool
      ~Loader.construct_yaml_float
      ~Loader.construct_yaml_int
      ~Loader.construct_yaml_map
      ~Loader.construct_yaml_null
      ~Loader.construct_yaml_object
      ~Loader.construct_yaml_omap
      ~Loader.construct_yaml_pairs
      ~Loader.construct_yaml_seq
      ~Loader.construct_yaml_set
      ~Loader.construct_yaml_str
      ~Loader.construct_yaml_timestamp
      ~Loader.descend_resolver
      ~Loader.determine_encoding
      ~Loader.dispose
      ~Loader.fetch_alias
      ~Loader.fetch_anchor
      ~Loader.fetch_block_entry
      ~Loader.fetch_block_scalar
      ~Loader.fetch_directive
      ~Loader.fetch_document_end
      ~Loader.fetch_document_indicator
      ~Loader.fetch_document_start
      ~Loader.fetch_double
      ~Loader.fetch_flow_collection_end
      ~Loader.fetch_flow_collection_start
      ~Loader.fetch_flow_entry
      ~Loader.fetch_flow_mapping_end
      ~Loader.fetch_flow_mapping_start
      ~Loader.fetch_flow_scalar
      ~Loader.fetch_flow_sequence_end
      ~Loader.fetch_flow_sequence_start
      ~Loader.fetch_folded
      ~Loader.fetch_key
      ~Loader.fetch_literal
      ~Loader.fetch_more_tokens
      ~Loader.fetch_plain
      ~Loader.fetch_single
      ~Loader.fetch_stream_end
      ~Loader.fetch_stream_start
      ~Loader.fetch_tag
      ~Loader.fetch_value
      ~Loader.flatten_mapping
      ~Loader.forward
      ~Loader.get_data
      ~Loader.get_event
      ~Loader.get_mark
      ~Loader.get_node
      ~Loader.get_single_data
      ~Loader.get_single_node
      ~Loader.get_token
      ~Loader.include
      ~Loader.need_more_tokens
      ~Loader.next_possible_simple_key
      ~Loader.parse_block_mapping_first_key
      ~Loader.parse_block_mapping_key
      ~Loader.parse_block_mapping_value
      ~Loader.parse_block_node
      ~Loader.parse_block_node_or_indentless_sequence
      ~Loader.parse_block_sequence_entry
      ~Loader.parse_block_sequence_first_entry
      ~Loader.parse_document_content
      ~Loader.parse_document_end
      ~Loader.parse_document_start
      ~Loader.parse_flow_mapping_empty_value
      ~Loader.parse_flow_mapping_first_key
      ~Loader.parse_flow_mapping_key
      ~Loader.parse_flow_mapping_value
      ~Loader.parse_flow_node
      ~Loader.parse_flow_sequence_entry
      ~Loader.parse_flow_sequence_entry_mapping_end
      ~Loader.parse_flow_sequence_entry_mapping_key
      ~Loader.parse_flow_sequence_entry_mapping_value
      ~Loader.parse_flow_sequence_first_entry
      ~Loader.parse_implicit_document_start
      ~Loader.parse_indentless_sequence_entry
      ~Loader.parse_node
      ~Loader.parse_stream_start
      ~Loader.peek
      ~Loader.peek_event
      ~Loader.peek_token
      ~Loader.prefix
      ~Loader.process_directives
      ~Loader.process_empty_scalar
      ~Loader.remove_possible_simple_key
      ~Loader.resolve
      ~Loader.save_possible_simple_key
      ~Loader.scan_anchor
      ~Loader.scan_block_scalar
      ~Loader.scan_block_scalar_breaks
      ~Loader.scan_block_scalar_ignored_line
      ~Loader.scan_block_scalar_indentation
      ~Loader.scan_block_scalar_indicators
      ~Loader.scan_directive
      ~Loader.scan_directive_ignored_line
      ~Loader.scan_directive_name
      ~Loader.scan_flow_scalar
      ~Loader.scan_flow_scalar_breaks
      ~Loader.scan_flow_scalar_non_spaces
      ~Loader.scan_flow_scalar_spaces
      ~Loader.scan_line_break
      ~Loader.scan_plain
      ~Loader.scan_plain_spaces
      ~Loader.scan_tag
      ~Loader.scan_tag_directive_handle
      ~Loader.scan_tag_directive_prefix
      ~Loader.scan_tag_directive_value
      ~Loader.scan_tag_handle
      ~Loader.scan_tag_uri
      ~Loader.scan_to_next_token
      ~Loader.scan_uri_escapes
      ~Loader.scan_yaml_directive_number
      ~Loader.scan_yaml_directive_value
      ~Loader.stale_possible_simple_keys
      ~Loader.unwind_indent
      ~Loader.update
      ~Loader.update_raw
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Loader.DEFAULT_MAPPING_TAG
      ~Loader.DEFAULT_SCALAR_TAG
      ~Loader.DEFAULT_SEQUENCE_TAG
      ~Loader.DEFAULT_TAGS
      ~Loader.ESCAPE_CODES
      ~Loader.ESCAPE_REPLACEMENTS
      ~Loader.NON_PRINTABLE
      ~Loader.bool_values
      ~Loader.inf_value
      ~Loader.nan_value
      ~Loader.timestamp_regexp
      ~Loader.yaml_constructors
      ~Loader.yaml_implicit_resolvers
      ~Loader.yaml_multi_constructors
      ~Loader.yaml_path_resolvers
   
   