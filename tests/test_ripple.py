"""
Tests for Material-style ripple effect on primary buttons.

Behavior: Clicking `.btn-primary` shows white ripple expanding from click point
"""

import pytest
from playwright.sync_api import expect


@pytest.fixture
def page_with_server(page):
    """Serve the static index.html file."""
    page.goto("file:///Users/andrew/Documents/Projects/DJ Playlist Optimiser/static/index.html")
    return page


class TestRippleEffect:
    """
    Test class for Material-style ripple effect behavior.
    
    Ubiquitous Language Verification:
    - Uses 'btn-primary' class selector (presentation layer term, no domain drift)
    """

    def test_ripple_element_created_on_primary_button_click(self, page_with_server):
        """
        Verify clicking a .btn-primary creates a ripple element.

        Expected behavior:
        - Click on .btn-primary button
        - A span element with class 'ripple' should be appended to the button
        - The ripple should have CSS animation applied
        """
        page = page_with_server
        
        # Find the Sort Playlist button (btn-primary)
        button = page.locator("#btn-sort")
        
        # Verify button exists and is visible
        expect(button).to_be_visible()
        expect(button).to_have_class("btn-primary")
        
        # Click the button
        button.click()
        
        # Verify ripple element was created
        ripple = button.locator("span.ripple")
        expect(ripple).to_have_count(1)

    def test_multiple_rapid_clicks_create_overlapping_ripples(self, page_with_server):
        """
        Verify multiple rapid clicks on .btn-primary create overlapping ripples.

        Expected behavior:
        - Click on .btn-primary button 3 times rapidly
        - 3 span elements with class 'ripple' should be appended to the button
        - All ripples should coexist (overlapping allowed)
        """
        page = page_with_server

        # Find the Sort Playlist button (btn-primary)
        button = page.locator("#btn-sort")

        # Verify button exists and is visible
        expect(button).to_be_visible()

        # Click the button 3 times rapidly
        for _ in range(3):
            button.click()

        # Verify 3 ripple elements were created
        ripples = button.locator("span.ripple")
        expect(ripples).to_have_count(3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
